use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::engine::DeviceConfig;
use crate::AppState;

#[derive(serde::Serialize)]
pub struct HardwareInfo {
    pub gpu_name: String,
    pub cuda_available: bool,
    pub directml_available: bool,
    pub current_device: String,
    /// Per-adapter vendor classification (S42, for runtime-pack recommendation).
    /// Vendor comes from PNPDeviceID's VEN_xxxx — NEVER from WMI AdapterRAM (a lying
    /// uint32: this dev box reports the 3080 Ti as 4 GB) and never from name heuristics.
    pub gpus: Vec<GpuAdapter>,
    /// Which runtime-pack variant this machine should default to
    /// ("nv-cu130" | "amd" | "xpu" | "cpu") — the user can always override.
    pub recommended_variant: String,
}

#[derive(serde::Serialize, Clone)]
pub struct GpuAdapter {
    pub name: String,
    /// "nvidia" | "amd" | "intel" | "other"
    pub vendor: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub device: DeviceConfig,
    /// User-set data root for the BIG growable files (models + cache). Empty/None → app_dir/data (next
    /// to the program, NOT C: AppData — those files reach tens of GB). See `resolve_data_dir`.
    #[serde(default)]
    pub data_dir: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::Auto,
            data_dir: None,
        }
    }
}

#[tauri::command]
pub fn get_hardware_info(state: State<'_, Arc<AppState>>) -> Result<HardwareInfo, String> {
    let current = state.inference.engine.device();
    let current_str = match &current {
        DeviceConfig::Cpu => "cpu".to_string(),
        DeviceConfig::DirectMl { .. } => "directml".to_string(),
        DeviceConfig::Cuda { .. } => "cuda".to_string(),
        DeviceConfig::Auto => "auto".to_string(),
    };

    let gpus = query_gpu_adapters();
    let gpu_name = if gpus.is_empty() {
        "Unknown GPU".to_string()
    } else {
        gpus.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")
    };
    Ok(HardwareInfo {
        gpu_name,
        // Vendor-guarded (S64c audit): the self-downloaded runtime/cuda DLLs satisfy the PATH probe
        // even on a box whose NVIDIA card is gone (migrated data dir) — the badge must track the GPU.
        cuda_available: gpus.iter().any(|g| g.vendor == "nvidia") && is_cuda_available(),
        directml_available: cfg!(windows),
        current_device: current_str,
        recommended_variant: recommend_variant(&gpus).to_string(),
        gpus,
    })
}

/// Default runtime-pack variant for this machine. NVIDIA wins over everything (the
/// only fully-supported training path); AMD over Intel. iGPU-vs-dGPU is deliberately
/// NOT guessed — the pick is only a DEFAULT and the UI lets the user override
/// (Pinokio's silent wrong-variant installs are the anti-pattern we're avoiding).
fn recommend_variant(gpus: &[GpuAdapter]) -> &'static str {
    if gpus.iter().any(|g| g.vendor == "nvidia") {
        "nv-cu130"
    } else if gpus.iter().any(|g| g.vendor == "amd") {
        "amd"
    } else if gpus.iter().any(|g| g.vendor == "intel") {
        "xpu"
    } else {
        "cpu"
    }
}

/// Enumerate video adapters with PCI vendor ids via WMI. One query serves both the
/// display string and the vendor classification (single source — replaces the old
/// name-only `detect_gpu_name`).
pub(crate) fn query_gpu_adapters() -> Vec<GpuAdapter> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress",
            ])
            .creation_flags(crate::util::CREATE_NO_WINDOW)
            .output();
        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                // ConvertTo-Json yields an OBJECT for one adapter, an ARRAY for several.
                let items: Vec<&serde_json::Value> = match &val {
                    serde_json::Value::Array(a) => a.iter().collect(),
                    other => vec![other],
                };
                let adapters: Vec<GpuAdapter> = items
                    .into_iter()
                    .filter_map(|item| {
                        let name = item.get("Name")?.as_str()?.trim().to_string();
                        let pnp = item.get("PNPDeviceID").and_then(|v| v.as_str()).unwrap_or("");
                        let vendor = if pnp.contains("VEN_10DE") {
                            "nvidia"
                        } else if pnp.contains("VEN_1002") {
                            "amd"
                        } else if pnp.contains("VEN_8086") {
                            "intel"
                        } else {
                            "other"
                        };
                        Some(GpuAdapter { name, vendor: vendor.to_string() })
                    })
                    .collect();
                if !adapters.is_empty() {
                    return adapters;
                }
            }
        }
    }
    Vec::new()
}

/// Max NVIDIA compute capability across installed NVIDIA GPUs, via nvidia-smi
/// (authoritative — WMI/PNPDeviceID can't report it). `None` when nvidia-smi is
/// absent or unreadable (no driver, or a non-NVIDIA box): callers treat `None` as
/// "undetermined → do not architecture-gate" (fail open, envtest is the real gate).
#[cfg(windows)]
pub(crate) fn nvidia_max_compute_cap() -> Option<f32> {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let max = text
        .lines()
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .fold(f32::NAN, f32::max); // f32::max ignores NaN → seeds cleanly, stays NaN if no rows
    if max.is_nan() {
        None
    } else {
        Some(max)
    }
}

#[cfg(not(windows))]
pub(crate) fn nvidia_max_compute_cap() -> Option<f32> {
    None
}

/// Whether THIS machine's hardware can run a given runtime-pack VARIANT — the gate for
/// which download entries the settings UI offers (only expose packs the user can actually
/// use; a fresh box always sees CPU). Vendor comes from PNPDeviceID. The NVIDIA pack
/// ADDITIONALLY needs an sm_75+ card (compute cap ≥ 7.5): torch cu130's fatbin floor is
/// sm_75, so a GTX 10-series / Pascal card can't run it and must not be offered the pack.
/// AMD/Intel are gated on vendor presence ONLY (experimental tier — the on-device envtest
/// is the true capability gate, and robust RDNA/Arc arch detection needs tooling we don't
/// bundle). An UNDETERMINED NVIDIA compute cap (nvidia-smi absent) fails OPEN so a valid
/// RTX user is never hidden. NB: LOCAL-FILE install is deliberately NOT gated by this.
pub(crate) fn variant_supported(variant: &str, gpus: &[GpuAdapter], nv_cc: Option<f32>) -> bool {
    let has = |v: &str| gpus.iter().any(|g| g.vendor == v);
    match variant {
        "cpu" => true,
        "nv-cu130" => has("nvidia") && nv_cc.map_or(true, |cc| cc >= 7.5),
        "amd" => has("amd"),
        "xpu" => has("intel"),
        _ => false,
    }
}

#[tauri::command]
pub fn set_device_preference(
    state: State<'_, Arc<AppState>>,
    device: String,
) -> Result<(), String> {
    let config = match device.as_str() {
        "cuda" => DeviceConfig::Cuda { device_id: 0 },
        "directml" => DeviceConfig::DirectMl { device_id: 0 },
        "cpu" => DeviceConfig::Cpu,
        _ => DeviceConfig::Auto,
    };

    state.inference.engine.set_device(config.clone());

    // Persist — load-then-update so we never clobber the rest of the config (esp. data_dir).
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.device = config;
    if let Err(e) = save_config(&state.app_dir, &cfg) {
        tracing::warn!("Failed to save config: {}", e);
    }

    Ok(())
}

#[tauri::command]
pub fn get_device_preference(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let current = state.inference.engine.device();
    Ok(match current {
        DeviceConfig::Cpu => "cpu".to_string(),
        DeviceConfig::DirectMl { .. } => "directml".to_string(),
        DeviceConfig::Cuda { .. } => "cuda".to_string(),
        DeviceConfig::Auto => "auto".to_string(),
    })
}

pub fn load_and_apply_config(state: &AppState) {
    // Logging rules (S22 + S42): state FACTS, not the fallback chain — which ORT
    // build this process committed is already known (ORT_LOADED_BUILD), and the
    // per-inference "ONNX device=..." lines remain the truth source for what each
    // run executes on. Logs are English/standard format (Chinese belongs to the
    // user-facing error strings, not tracing). NB: an absent config.json MEANS the
    // preference IS Auto (the default is simply never written to disk) — the old
    // wording ("No config found") read like breakage and was mistaken for a CUDA
    // regression in the field.
    let build = crate::ORT_LOADED_BUILD.get().map(|s| s.as_str()).unwrap_or("?");
    if let Some(cfg) = load_config(&state.app_dir) {
        tracing::info!(
            "device preference: {:?} (config.json); ORT build loaded: {}; per-run EP is logged as \"ONNX device=...\"",
            cfg.device,
            build
        );
        state.inference.engine.set_device(cfg.device);
    } else {
        tracing::info!(
            "device preference: Auto (default; config.json is only written once changed in Settings); ORT build loaded: {}; per-run EP is logged as \"ONNX device=...\"",
            build
        );
    }
}

fn config_path(app_dir: &std::path::Path) -> std::path::PathBuf {
    app_dir.join("config.json")
}

fn save_config(app_dir: &std::path::Path, cfg: &AppConfig) -> std::io::Result<()> {
    let path = config_path(app_dir);
    let json = serde_json::to_string_pretty(cfg).unwrap_or_default();
    // Temp + rename so a crash mid-write can't truncate config.json (losing device pref + data_dir).
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)
}

fn load_config(app_dir: &std::path::Path) -> Option<AppConfig> {
    let path = config_path(app_dir);
    let content = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&content) {
        Ok(cfg) => Some(cfg),
        Err(e) => {
            // A corrupt config silently falling back to defaults would look like lost settings.
            tracing::warn!("config.json exists but failed to parse ({}); using defaults", e);
            None
        }
    }
}

/// S64 portability: the data-dir override in config.json is an ABSOLUTE user-chosen path (the one
/// sanctioned absolute reference) — when its target vanishes (drive unplugged, dir deleted, install
/// copied to another machine) the old behavior was a SILENT empty library (models/dictionaries/
/// runtimes all "gone", zero warnings). This records what happened for the settings UI + a startup
/// toast; set at most once, at startup resolution.
#[derive(serde::Serialize, Clone)]
pub struct DataDirIssue {
    /// The configured (missing) override path.
    pub configured: String,
    /// The directory actually used this session.
    pub effective: String,
    /// true = override unusable (drive gone) → default next to the program; false = recreated empty.
    pub fell_back: bool,
}

pub static DATA_DIR_ISSUE: std::sync::OnceLock<DataDirIssue> = std::sync::OnceLock::new();

/// Startup warning for the frontend (null = the data dir resolved normally).
#[tauri::command]
pub fn get_data_dir_issue() -> Option<DataDirIssue> {
    DATA_DIR_ISSUE.get().cloned()
}

/// Data root for the big growable files (models + cache). User-set in config.json's `data_dir`; else
/// `app_dir/data` — NEXT TO THE PROGRAM, never C: AppData (those files reach tens of GB). Derived at
/// startup; changing it takes effect on restart. A configured-but-missing override is recreated on
/// its drive when possible (user intent wins), else falls back to the default — either way LOUDLY
/// (DATA_DIR_ISSUE), never a silent empty library.
pub fn resolve_data_dir(app_dir: &std::path::Path) -> std::path::PathBuf {
    if let Some(cfg) = load_config(app_dir) {
        if let Some(d) = cfg.data_dir {
            let d = d.trim();
            if !d.is_empty() {
                let p = std::path::PathBuf::from(d);
                if p.is_dir() {
                    return p;
                }
                if std::fs::create_dir_all(&p).is_ok() {
                    tracing::warn!("configured data_dir {} was missing — recreated (empty)", d);
                    let _ = DATA_DIR_ISSUE.set(DataDirIssue {
                        configured: d.to_string(),
                        effective: p.to_string_lossy().to_string(),
                        fell_back: false,
                    });
                    return p;
                }
                let fallback = app_dir.join("data");
                tracing::warn!(
                    "configured data_dir {} is unavailable — falling back to {}",
                    d,
                    fallback.display()
                );
                let _ = DATA_DIR_ISSUE.set(DataDirIssue {
                    configured: d.to_string(),
                    effective: fallback.to_string_lossy().to_string(),
                    fell_back: true,
                });
                return fallback;
            }
        }
    }
    app_dir.join("data")
}

/// The data root ACTUALLY in use this session — parent of cache_dir (cache_dir = data_root/cache,
/// models = data_root/models). May differ from `resolve_data_dir`: startup can pick the legacy
/// AppData fallback for upgraders (see lib.rs setup).
fn effective_data_root(state: &AppState) -> &std::path::Path {
    state.cache_dir.parent().unwrap_or(state.cache_dir.as_path())
}

/// Current data dir (for the settings UI).
#[tauri::command]
pub fn get_data_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    Ok(effective_data_root(&state).to_string_lossy().to_string())
}

/// Recursively copy a directory's contents into `dst` (creating it). Cross-drive safe (copy, not rename).
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// One-click migrate: copy the CURRENT models + cache into `new_dir`, then persist it as the data dir.
/// Takes effect on restart. Old data is LEFT in place (rollback = revert the setting / don't restart);
/// the user deletes the old copy manually once the new location is confirmed working.
#[tauri::command]
pub async fn migrate_data_dir(state: State<'_, Arc<AppState>>, new_dir: String) -> Result<(), String> {
    let new = std::path::PathBuf::from(new_dir.trim());
    if new.as_os_str().is_empty() {
        return Err("Empty target directory".into());
    }
    // S61: a live training run writes checkpoints/features mid-copy — the migrated tree would be
    // torn (and the workspace copy below is exactly what a running trainer mutates).
    if state.training.is_active() {
        return Err("TRAINING_ACTIVE".into());
    }
    let data_root = effective_data_root(&state).to_path_buf();
    let target = new.clone();
    // The copy reaches tens of GB — run it off the event loop so the UI stays responsive.
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        std::fs::create_dir_all(&target).map_err(|e| format!("Create target: {e}"))?;
        // Refuse a target nested inside the data root (or vice versa) — copying a tree into itself
        // recurses forever.
        let canon_target = std::fs::canonicalize(&target).map_err(|e| format!("Resolve target: {e}"))?;
        let canon_root = std::fs::canonicalize(&data_root).unwrap_or_else(|_| data_root.clone());
        if canon_target.starts_with(&canon_root) || canon_root.starts_with(&canon_target) {
            return Err("Target directory overlaps the current data directory".into());
        }
        copy_dir_all(&data_root.join("models"), &target.join("models")).map_err(|e| format!("Copy models: {e}"))?;
        copy_dir_all(&data_root.join("cache"), &target.join("cache")).map_err(|e| format!("Copy cache: {e}"))?;
        // ② S58: the stage1 G2P dictionaries live under <data_root>/dictionaries — leaving them behind
        // would fake-OOV every zh/en/de/fr/es/it lyric after a migration (audit MAJOR).
        let dicts_src = data_root.join("dictionaries");
        if dicts_src.exists() {
            copy_dir_all(&dicts_src, &target.join("dictionaries")).map_err(|e| format!("Copy dictionaries: {e}"))?;
        }
        // Runtime packs (S42) live under <data_root>/runtimes and must MOVE WITH the
        // data dir — lib.rs roots pyenv on the resolved data dir, so leaving them
        // behind would make every installed pack "vanish" after migration (and strand
        // gigabytes on the old drive with no UI to reclaim them). `.staging` (torn
        // installs / resumable part files) is transient — skip it.
        let runtimes_src = data_root.join("runtimes");
        if runtimes_src.exists() {
            let runtimes_dst = target.join("runtimes");
            std::fs::create_dir_all(&runtimes_dst).map_err(|e| format!("Create runtimes: {e}"))?;
            for entry in std::fs::read_dir(&runtimes_src).map_err(|e| format!("Read runtimes: {e}"))?.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with('.') {
                    continue;
                }
                copy_dir_all(&entry.path(), &runtimes_dst.join(&name))
                    .map_err(|e| format!("Copy runtimes/{}: {e}", name.to_string_lossy()))?;
            }
        }
        // S61 (recon gap): training WORKSPACES live under <data_root>/training and resolve off the
        // SAME data dir (commands/training.rs data_root) — not copying them silently stranded every
        // checkpoint + dataset on the old drive while 续训/共享池 resolved against the NEW (empty)
        // tree after restart. GBs, but losing training progress is worse than a longer copy.
        copy_dir_all(&data_root.join("training"), &target.join("training"))
            .map_err(|e| format!("Copy training: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("Copy task failed: {e}"))??;
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.data_dir = Some(new.to_string_lossy().to_string());
    save_config(&state.app_dir, &cfg).map_err(|e| format!("Save config: {e}"))?;
    tracing::info!("Migrated data dir → {} (restart to apply)", new.display());
    Ok(())
}

/// Whether CUDA is ACTUALLY usable, not just "files downloaded". Verifies that the CUDA ORT build is
/// present AND that the CUDA major it was built for (read from providers_cuda.dll's imports) matches a
/// cudart + cuDNN actually resolvable on this machine. This is what stops the old false "Ready" when a
/// CUDA-11-built ORT (1.21.x) sat on a CUDA-12 system — it now correctly reports NOT ready.
#[tauri::command]
pub fn is_cuda_runtime_ready(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    let cuda_dir = state.app_dir.join("runtime").join("ort").join("cuda");
    let ort_cuda_dll = cuda_dir.join("onnxruntime.dll");
    let providers = cuda_dir.join("onnxruntime_providers_cuda.dll");
    if !ort_cuda_dll.exists() || !providers.exists() {
        return Ok(false);
    }
    // Which CUDA major does this build actually need? (1.21.x wrongly needs 11 → unusable on a 12 box.)
    let major = cuda_build_major(&providers).unwrap_or(0);
    if major < 12 {
        return Ok(false); // CUDA 11 build (or unreadable) — treat as not ready
    }
    Ok(cuda_provider_deps_resolvable(&state.app_dir))
}

/// THE provider-dependency check (S64c): the FULL import set scanned from the 1.24.4
/// providers_cuda.dll, each resolvable from OUR runtime/cuda (self-contained download), PATH, or
/// CUDA_PATH (Toolkit users). Shared by is_cuda_runtime_ready AND lib.rs' Auto build pick — a
/// PARTIAL install must never flip Auto onto the CUDA build (it has no DirectML provider).
pub(crate) fn cuda_provider_deps_resolvable(app_dir: &std::path::Path) -> bool {
    const DEPS: [&str; 5] = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "cudnn64_9.dll",
    ];
    let cuda_dir = app_dir.join("runtime").join("cuda");
    DEPS.iter().all(|d| dll_on_path_or_dir(d, &cuda_dir))
}

/// Scan a providers_cuda.dll for its imported `cudart64_NNN.dll` string to learn the CUDA MAJOR it was
/// built against (110 → 11, 12 → 12, 118 → 11). Reads the whole DLL once; fine for an on-demand check.
fn cuda_build_major(providers_cuda: &std::path::Path) -> Option<u32> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    // Cache keyed by (path, mtime, len) so repeated Settings opens don't re-read the DLL, while a
    // re-download replacing it in-session (do_download_cuda_runtime) is picked up without a restart.
    type CacheKey = (std::path::PathBuf, Option<std::time::SystemTime>, u64);
    static CACHE: Mutex<Option<HashMap<CacheKey, Option<u32>>>> = Mutex::new(None);
    let meta = std::fs::metadata(providers_cuda).ok();
    let key: CacheKey = (
        providers_cuda.to_path_buf(),
        meta.as_ref().and_then(|m| m.modified().ok()),
        meta.as_ref().map(|m| m.len()).unwrap_or(0),
    );
    if let Some(m) = CACHE.lock().unwrap().as_ref() {
        if let Some(v) = m.get(&key) {
            return *v;
        }
    }
    let result = scan_cuda_major(providers_cuda);
    CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    result
}

fn scan_cuda_major(providers_cuda: &std::path::Path) -> Option<u32> {
    use std::io::Read;
    // The "cudart64_NNN.dll" import string lives near the PE header, not in the hundreds-of-MB CUDA
    // kernel blob — read only the first 64MB instead of slurping the whole DLL into RAM.
    let mut data = Vec::new();
    std::fs::File::open(providers_cuda)
        .ok()?
        .take(64 * 1024 * 1024)
        .read_to_end(&mut data)
        .ok()?;
    let needle = b"cudart64_";
    let mut i = 0usize;
    while i + needle.len() + 1 < data.len() {
        if &data[i..i + needle.len()] == needle {
            let mut j = i + needle.len();
            let mut digits = String::new();
            while j < data.len() && data[j].is_ascii_digit() && digits.len() < 4 {
                digits.push(data[j] as char);
                j += 1;
            }
            if let Ok(n) = digits.parse::<u32>() {
                return Some(if n >= 100 { n / 10 } else { n });
            }
        }
        i += 1;
    }
    None
}

/// True if `name` is found on PATH or in the system CUDA Toolkit bin (CUDA_PATH may not be on PATH here).
fn dll_on_path(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        if std::env::split_paths(&path).any(|d| d.join(name).exists()) {
            return true;
        }
    }
    if let Ok(cuda) = std::env::var("CUDA_PATH") {
        if std::path::Path::new(&cuda).join("bin").join(name).exists() {
            return true;
        }
    }
    false
}

fn dll_on_path_or_dir(name: &str, extra: &std::path::Path) -> bool {
    extra.join(name).exists() || dll_on_path(name)
}

/// Download CUDA ORT DLLs + cuDNN DLLs for CUDA EP support.
/// Emits `cuda-download-progress` events with {stage, progress, message}.
#[tauri::command]
pub async fn download_cuda_runtime(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    // S64c: the download is now fully self-contained (cudart/cublas/cufft/cudnn all fetched from
    // NVIDIA's official PyPI redistributables — no CUDA Toolkit needed, which beta testers proved
    // nobody has). The one hard requirement left is an NVIDIA GPU + its driver. FAIL-OPEN on an
    // EMPTY probe (WMI/PowerShell failure = undetermined, the variant_supported convention) —
    // refuse only on a POSITIVE non-NVIDIA determination.
    let gpus = query_gpu_adapters();
    if !gpus.is_empty() && !gpus.iter().any(|g| g.vendor == "nvidia") {
        return Err("CUDA_GPU_REQUIRED".to_string());
    }
    // Single-flight (S64c audit): begin_task is a refcount for the close-flow listing, not a mutex —
    // a remounted Settings panel re-enables the button mid-download, and a second click would run
    // two concurrent downloaders over the same files.
    if state.task_active("cuda_download") {
        return Err("CUDA_DOWNLOAD_BUSY".to_string());
    }
    let _task = state.begin_task("cuda_download"); // listed in the close-flow's in-progress warning
    let app_dir = state.app_dir.clone();
    let handle = app_handle.clone();

    let result = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            do_download_cuda_runtime(&app_dir, &handle).await
        })
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?;

    // Surface the outcome into the tracing pipeline (log panel + file) — a failed download used to be
    // invisible there (only shown under the button), which is exactly what the user hit.
    match &result {
        Ok(()) => tracing::info!("CUDA runtime download complete"),
        Err(e) => tracing::error!("CUDA runtime download failed: {}", e),
    }
    result.map_err(|e| e.to_string())
}

async fn do_download_cuda_runtime(
    app_dir: &std::path::Path,
    handle: &tauri::AppHandle,
) -> crate::Result<()> {

    // code+label ride along for i18n (frontend maps code → localized line, label = proper noun;
    // message stays as the raw-English fallback — the S62 pyenv structured-progress pattern).
    let emit = |stage: &str, progress: f32, code: &str, label: &str, msg: &str| {
        let _ = handle.emit("cuda-download-progress", serde_json::json!({
            "stage": stage, "progress": progress, "code": code, "label": label, "message": msg,
        }));
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| crate::UtaiError::Audio(format!("HTTP client: {}", e)))?;

    // ── Stage 1: CUDA ORT DLLs from NuGet ──
    // 1.24.4 MUST match the ORT API version the `ort` crate (2.0-rc.12) targets — API 24 — AND the
    // bundled DirectML build (1.24.4). A mismatched CUDA build (e.g. 1.20.1 = API 20) makes ort's
    // init_from of the CUDA build DEADLOCK (ort calls API-24 ABI against an API-20 DLL). 1.24.4's
    // providers_cuda imports cudart64_12 / cublas64_12+Lt / cufft64_11 / cudnn64_9.
    // AVOID 1.21.x (mis-built against CUDA 11). Gpu.Windows has the actual DLLs.
    emit("ort", 0.0, "CUDA_DL_DOWNLOADING", "CUDA ORT", "Downloading CUDA ORT runtime...");
    let ort_cuda_dir = app_dir.join("runtime").join("ort").join("cuda");
    let ort_url = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Windows/1.24.4";
    let ort_nupkg = app_dir.join("runtime").join("ort_gpu.nupkg.zip");

    // Download FIRST, wipe after (S64c audit): the old wipe-then-download order destroyed a good
    // install before the replacement bytes were secured — a failed retry left NOTHING.
    download_file(&client, ort_url, &ort_nupkg, |p| {
        emit("ort", p * 0.2, "CUDA_DL_DOWNLOADING", "CUDA ORT", "Downloading CUDA ORT...")
    })
    .await?;
    emit("ort", 0.2, "CUDA_DL_EXTRACTING", "CUDA ORT", "Extracting CUDA ORT DLLs...");
    // Wipe any previous (possibly wrong-CUDA) DLLs so the extraction REPLACES them cleanly; an
    // extraction failure here self-heals (no skip guard — the next attempt wipes and re-extracts).
    let _ = std::fs::remove_dir_all(&ort_cuda_dir);
    std::fs::create_dir_all(&ort_cuda_dir)?;
    crate::util::extract_zip_dlls(&ort_nupkg, &ort_cuda_dir, |n| n.starts_with("runtimes/win-x64/native"))?;
    let _ = std::fs::remove_file(&ort_nupkg);

    // ── Stage 2 (S64c): the provider's FULL import set from NVIDIA's official PyPI redistributables —
    //    cudart64_12 / cublas64_12+Lt / cufft64_11 / cudnn64_9 (the exact list scanned from the 1.24.4
    //    providers_cuda.dll). No CUDA Toolkit install needed; runtime/cuda sits FIRST in
    //    setup_cuda_dll_paths' search dirs, so our copies also win over a wrong-major Toolkit (e.g. 13).
    //    Each lane SKIPS when its DLL is already present (runtime/cuda is kept across re-downloads;
    //    a flaky/blocked network must not fail an otherwise-complete install). ──
    let cuda_dir = app_dir.join("runtime").join("cuda");
    std::fs::create_dir_all(&cuda_dir)?;
    struct Wheel {
        guard: &'static str,  // presence of this DLL skips the lane
        url: &'static str,    // pinned pythonhosted wheel (versions chosen for the cu12 family)
        filter: &'static str, // wheel-internal bin dir holding the DLLs
        label: &'static str,
        p0: f32,
        p1: f32,
    }
    const WHEELS: [Wheel; 4] = [
        Wheel { guard: "cudart64_12.dll", url: "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl", filter: "nvidia/cuda_runtime/bin", label: "CUDA runtime", p0: 0.25, p1: 0.28 },
        Wheel { guard: "cublas64_12.dll", url: "https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl", filter: "nvidia/cublas/bin", label: "cuBLAS", p0: 0.28, p1: 0.55 },
        Wheel { guard: "cufft64_11.dll", url: "https://files.pythonhosted.org/packages/20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl", filter: "nvidia/cufft/bin", label: "cuFFT", p0: 0.55, p1: 0.65 },
        Wheel { guard: "cudnn64_9.dll", url: "https://files.pythonhosted.org/packages/f2/a4/045f8d0ce6b99726d88e76bbb8ee147123f55e80111d89262762d8149abb/nvidia_cudnn_cu12-9.22.0.52-py3-none-win_amd64.whl", filter: "nvidia/cudnn/bin", label: "cuDNN", p0: 0.65, p1: 0.93 },
    ];
    for w in &WHEELS {
        if cuda_dir.join(w.guard).exists() {
            emit("cuda", w.p1, "CUDA_DL_SKIP", w.label, &format!("{} already present — skipping", w.label));
            tracing::info!("CUDA download: {} already present, skipping", w.label);
            continue;
        }
        emit("cuda", w.p0, "CUDA_DL_DOWNLOADING", w.label, &format!("Downloading {}...", w.label));
        let tmp = app_dir.join("runtime").join(format!("{}.whl.zip", w.guard));
        if let Err(e) = download_file(&client, w.url, &tmp, |p| {
            emit("cuda", w.p0 + p * (w.p1 - w.p0) * 0.9, "CUDA_DL_DOWNLOADING", w.label, &format!("Downloading {}...", w.label))
        })
        .await
        {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        emit("cuda", w.p0 + (w.p1 - w.p0) * 0.9, "CUDA_DL_EXTRACTING", w.label, &format!("Extracting {}...", w.label));
        // ATOMIC placement (S64c audit MAJOR): extract into a per-lane staging dir, then rename each
        // DLL into runtime/cuda with the GUARD LAST — guard presence ⇒ lane complete. The naive
        // in-place extraction could die between a wheel's DLLs (cublas64 lands, cublasLt doesn't) and
        // the guard skip would then wedge the lane FOREVER (ready-check false, every retry "done");
        // worse, a torn cuDNN (guard fine, engine sub-DLLs missing) read as READY and only exploded
        // at the first Conv after restart (the S60c failure class).
        let stage_dir = app_dir.join("runtime").join(format!("{}.extract", w.guard));
        let _ = std::fs::remove_dir_all(&stage_dir);
        std::fs::create_dir_all(&stage_dir)?;
        let placed = (|| -> crate::Result<()> {
            crate::util::extract_zip_dlls(&tmp, &stage_dir, |n| n.contains(w.filter))?;
            let mut names: Vec<std::ffi::OsString> = std::fs::read_dir(&stage_dir)?
                .flatten()
                .map(|e| e.file_name())
                .collect();
            // Guard renames LAST — its presence must imply every sibling already moved.
            names.sort_by_key(|n| n.eq_ignore_ascii_case(w.guard));
            for name in names {
                let dest = cuda_dir.join(&name);
                let _ = std::fs::remove_file(&dest); // Windows rename refuses to overwrite
                std::fs::rename(stage_dir.join(&name), &dest)?;
            }
            Ok(())
        })();
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_dir_all(&stage_dir);
        placed?;
    }

    // Make the fresh runtime resolvable IN-SESSION (S64c audit): runtime/cuda may not have existed
    // at startup, so it never got onto PATH — is_cuda_available's probe would stay false until a
    // restart while the runtime row says Installed. Re-running setup is idempotent.
    crate::setup_cuda_dll_paths(app_dir);

    // ── Stage 3 (DEV BUILDS ONLY): copy next to the debug exe. In release this polluted the
    // install root with the four CUDA DLLs (S64b beta report) — the installed app loads from
    // runtime/ort/cuda directly and needs no exe-side copies. lib.rs setup sweeps old strays. ──
    emit("copy", 0.95, "CUDA_DL_FINALIZING", "", "Finalizing...");
    #[cfg(debug_assertions)]
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let target_debug = exe_dir;
            // Copy CUDA ORT DLLs next to exe for dev convenience
            for entry in std::fs::read_dir(&ort_cuda_dir).into_iter().flatten().flatten() {
                let name = entry.file_name();
                let dest = target_debug.join(&name);
                // Overwrite unconditionally — a stale wrong-CUDA copy here would shadow the new one.
                let _ = std::fs::copy(entry.path(), &dest);
            }
        }
    }

    emit("done", 1.0, "CUDA_DL_DONE", "", "CUDA runtime ready. Restart to activate.");
    tracing::info!("CUDA runtime download complete: ORT={}, cuDNN={}", ort_cuda_dir.display(), cuda_dir.display());
    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
    progress_cb: impl Fn(f32),
) -> crate::Result<()> {
    let r = download_file_inner(client, url, dest, progress_cb).await;
    if r.is_err() {
        // No resume support here (legacy helper) — a partial file is pure dead weight, and callers'
        // retry semantics assume a clean slate (S64c audit: failed wheel tmps stranded 100s of MB).
        let _ = std::fs::remove_file(dest);
    }
    r
}

async fn download_file_inner(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
    progress_cb: impl Fn(f32),
) -> crate::Result<()> {
    let resp = client.get(url).send().await
        .map_err(|e| crate::UtaiError::Audio(format!("Download failed: {}", e)))?;

    if !resp.status().is_success() {
        return Err(crate::UtaiError::Audio(format!("HTTP {}: {}", resp.status(), url)));
    }

    let total = resp.content_length().unwrap_or(0);
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut file = std::fs::File::create(dest)?;
    let mut downloaded: u64 = 0;

    use std::io::Write;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| crate::UtaiError::Audio(format!("Download stream: {}", e)))?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            progress_cb(downloaded as f32 / total as f32);
        }
    }
    Ok(())
}

// extract_nupkg_dlls / extract_wheel_dlls moved to crate::util::extract_zip_dlls
// (callers pass a starts_with / contains closure for the path match).

pub(crate) fn is_cuda_available() -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // S64c self-contained runtime: cudart lives in runtime/cuda, which setup_cuda_dll_paths put
        // on PATH before any caller runs — a plain PATH scan covers it (and any real Toolkit).
        if dll_on_path("cudart64_12.dll") {
            return true;
        }
        // Check CUDA toolkit's standard install location first (fast)
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let bin = std::path::Path::new(&cuda_path).join("bin");
            if bin.exists() {
                if let Ok(entries) = std::fs::read_dir(&bin) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_lowercase();
                        if name.starts_with("cudart64_") && name.ends_with(".dll") {
                            return true;
                        }
                    }
                }
            }
        }
        // Fallback: check if nvcc is on PATH (lightweight — just runs one command)
        if let Ok(output) = std::process::Command::new("where")
            .arg("nvcc.exe")
            .creation_flags(crate::util::CREATE_NO_WINDOW)
            .output()
        {
            if output.status.success() {
                return true;
            }
        }
    }
    false
}
