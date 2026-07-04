use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::{rvc, sovits, RvcOptions, SovitsOptions, SynthesisResult, VoiceBackendType};
use crate::models::{ModelConfig, ModelEntry};
use crate::AppState;

/// Per-node inference progress, emitted as the `voice-progress` event. The frontend workflow
/// engine listens during the run_rvc/run_sovits invoke and drives the node's progress bar,
/// filtering by node_id.
#[derive(Clone, serde::Serialize)]
struct VoiceProgress {
    node_id: String,
    progress: f32,
}

/// Build a progress callback that emits throttled `voice-progress` events (only on a ≥1% step,
/// plus the terminal 1.0) so a many-chunk RVC run doesn't spam the event bus.
fn progress_emitter(app: tauri::AppHandle, node_id: String) -> impl Fn(f32) {
    let last = AtomicU32::new(0);
    move |p: f32| {
        let pct = (p * 100.0).round() as u32;
        if p >= 1.0 || pct > last.load(Ordering::Relaxed) {
            last.store(pct, Ordering::Relaxed);
            let _ = app.emit(
                "voice-progress",
                VoiceProgress {
                    node_id: node_id.clone(),
                    progress: p,
                },
            );
        }
    }
}

// ─── aux model resolution (models_dir/aux/...) ───────────────────────────────

const AUX_CONTENTVEC_768: &str = "contentvec_768l12.onnx";
const AUX_CONTENTVEC_256: &str = "contentvec_256l9.onnx";
const AUX_RMVPE: &str = "rmvpe_e2e.onnx";
const AUX_RMVPE_MEL: &str = "rmvpe_mel_filters.npy";

/// models_dir/aux/<filename>, with a clear Chinese error naming the missing file + the
/// exact directory it must be placed in.
fn aux_path(state: &AppState, filename: &str, label: &str) -> Result<PathBuf, String> {
    let dir = state.models.models_dir().join("aux");
    let path = dir.join(filename);
    if !path.exists() {
        return Err(format!(
            "缺少{} {}，请将其放入 {}",
            label,
            filename,
            dir.display()
        ));
    }
    Ok(path)
}

/// ContentVec variant routing: vec768l12 → RVC v2 / SoVITS 4.1, vec256l9 → RVC v1 / SoVITS 4.0.
fn contentvec_for_dim(state: &AppState, dim: usize) -> Result<PathBuf, String> {
    match dim {
        768 => aux_path(state, AUX_CONTENTVEC_768, "内容特征模型"),
        256 => aux_path(state, AUX_CONTENTVEC_256, "内容特征模型"),
        other => Err(format!(
            "不支持的内容特征维度 {}（仅支持 256 / 768）——请检查模型配置 features_dim / speech_encoder",
            other
        )),
    }
}

/// Effective feature dim: speech_encoder wins when present (SoVITS sidecars), else
/// features_dim (RVC sidecars carry it directly).
fn features_dim(config: &ModelConfig) -> Result<usize, String> {
    if let Some(enc) = config.speech_encoder.as_deref() {
        return match enc {
            "vec768l12" => Ok(768),
            "vec256l9" => Ok(256),
            other => Err(format!(
                "不支持的 speech_encoder：{}（仅支持 vec768l12 / vec256l9）",
                other
            )),
        };
    }
    Ok(config.features_dim as usize)
}

/// inter_channels of the model's noise input, from the sidecar "noise" block when present
/// (converter writes {"rnd_input"/"noise_input": [1, C, "T"]}); 192 for every standard
/// RVC v1/v2 and SoVITS 4.x config.
fn noise_channels(config: &ModelConfig) -> usize {
    config
        .noise
        .as_ref()
        .and_then(|v| v.get("rnd_input").or_else(|| v.get("noise_input")))
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(192)
}

/// Sidecar "min_frames": the minimum T the exported graph accepts (final contract:
/// RVC 12 / SoVITS 6). Tolerant field — lives in ModelConfig.extra.
fn min_frames(config: &ModelConfig, default: usize) -> usize {
    config
        .extra
        .get("min_frames")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
        .max(1)
}

/// Whether the sidecar "inputs" array contains `input` (None when the sidecar predates
/// the converter rework and has no such array).
fn sidecar_has_input(entry: &ModelEntry, input: &str) -> Option<bool> {
    entry
        .config
        .inputs
        .as_ref()
        .and_then(|v| v.as_array())
        .map(|list| list.iter().any(|v| v.as_str() == Some(input)))
}

/// New-signature guard: the S35 converter ALWAYS writes an `inputs` array listing the graph
/// inputs. Proceed ONLY when that array is present AND contains the required new input. Both a
/// missing input (Some(false), old export WITH an inputs list) and a missing inputs array
/// (None, pre-rework sidecar that never wrote one) mean the ONNX predates the rework — fail with
/// an actionable message instead of a cryptic raw ORT "Invalid Feed Input Name" crash.
fn require_input(entry: &ModelEntry, input: &str) -> Result<(), String> {
    if sidecar_has_input(entry, input) != Some(true) {
        return Err(format!(
            "模型 '{}' 是旧版导出格式（缺少 {} 输入签名），请删除后重新导入以完成升级",
            entry.name, input
        ));
    }
    Ok(())
}

fn get_entry(state: &AppState, voice_name: &str) -> Result<ModelEntry, String> {
    state.models.get(voice_name).ok_or_else(|| {
        format!(
            "未找到模型 '{}'，请先在资源管理器中导入",
            voice_name
        )
    })
}

// ─── run_rvc ─────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn run_rvc(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    voice_name: String,
    model_path: String,
    audio_path: String,
    node_id: String,
    options: RvcOptions,
) -> Result<SynthesisResult, String> {
    let app = state.inner().clone();
    let entry = get_entry(&app, &voice_name)?;
    require_input(&entry, "rnd")?;

    let dim = entry.config.features_dim as usize; // RVC sidecars carry features_dim directly
    let nch = noise_channels(&entry.config);
    let min_t = min_frames(&entry.config, 12);
    let cv_path = contentvec_for_dim(&app, dim)?;
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "音高检测模型")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "音高检测滤波器")?;

    let path = PathBuf::from(&model_path);
    app.inference
        .load_voice(
            &voice_name,
            &path,
            VoiceBackendType::Rvc,
            entry.sample_rate,
            entry.index_path.as_ref(),
        )
        .map_err(|e| e.to_string())?;

    let cv_sid = app.inference.ensure_aux_loaded(&cv_path).map_err(|e| e.to_string())?;
    let rmvpe_sid = app.inference.ensure_aux_loaded(&rmvpe_path).map_err(|e| e.to_string())?;
    let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;
    let handle = app.inference.voice_handle(&voice_name).map_err(|e| e.to_string())?;

    let audio_buf =
        crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;

    // The pipeline is minutes of CPU+GPU work — keep it off the async runtime workers.
    let progress = progress_emitter(app_handle, node_id);
    tauri::async_runtime::spawn_blocking(move || {
        let model = rvc::RvcModel {
            engine: &app.inference.engine,
            voice_session: &handle.session_id,
            contentvec_session: &cv_sid,
            rmvpe_session: &rmvpe_sid,
            mel_filters: mel.as_ref(),
            index: handle.index.as_deref(),
            sample_rate: handle.sample_rate,
            features_dim: dim,
            noise_channels: nch,
            min_frames: min_t,
        };
        rvc::run_pipeline(&model, &audio_buf, &options, &progress).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("推理任务失败: {}", e))?
}

// ─── run_sovits ──────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn run_sovits(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    voice_name: String,
    model_path: String,
    audio_path: String,
    node_id: String,
    options: SovitsOptions,
) -> Result<SynthesisResult, String> {
    let app = state.inner().clone();
    let entry = get_entry(&app, &voice_name)?;
    require_input(&entry, "noise")?;

    let dim = features_dim(&entry.config)?;
    let nch = noise_channels(&entry.config);
    let hop_size = entry.config.hop_size.unwrap_or(512) as usize;
    if hop_size == 0 {
        return Err(format!("模型 '{}' 配置的 hop_size 为 0，无法推理", voice_name));
    }
    let min_t = min_frames(&entry.config, 6);
    // Feed vol IFF the exported graph HAS the input — the sidecar "inputs" array is the
    // authority (final contract); vol_embedding bool is the fallback for older sidecars.
    let vol_embedding = sidecar_has_input(&entry, "vol")
        .unwrap_or_else(|| entry.config.vol_embedding.unwrap_or(false));
    let unit_interpolate_mode = entry
        .config
        .unit_interpolate_mode
        .clone()
        .unwrap_or_else(|| "left".to_string());

    let cv_path = contentvec_for_dim(&app, dim)?;
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "音高检测模型")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "音高检测滤波器")?;

    let path = PathBuf::from(&model_path);
    app.inference
        .load_voice(
            &voice_name,
            &path,
            VoiceBackendType::SoVits,
            entry.sample_rate,
            None,
        )
        .map_err(|e| e.to_string())?;

    let cv_sid = app.inference.ensure_aux_loaded(&cv_path).map_err(|e| e.to_string())?;
    let rmvpe_sid = app.inference.ensure_aux_loaded(&rmvpe_path).map_err(|e| e.to_string())?;
    let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;
    let handle = app.inference.voice_handle(&voice_name).map_err(|e| e.to_string())?;

    // cluster 资产（converter\export_cluster.py 最终合约）：导入时落进 <stem>.cluster\ 子目录
    // （resolve_cluster_assets；多个 SoVITS 模型共用 sovits\ 目录，平铺 spk-id 会撞名）。
    //   特征检索：<speaker_id>.index_vectors.npy（spk2id 整数键，[N, dim]，优先，
    //             与原版 feature_retrieval 一致）
    //   kmeans： <speaker_name>.centers.npy（speaker 名字键，可能是中文；
    //             路径非法字符按 export_cluster 的 _safe_name 规则 →'_'，[K, dim]）
    // 兼容手动平铺在模型旁的旧摆法。
    let cluster = if options.cluster_ratio > 0.0 {
        let spk = options.speaker_id.unwrap_or(0);
        let parent = entry.path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let stem = entry.path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let cluster_dir = parent.join(format!("{}.cluster", stem));
        let model_dir = cluster_dir; // primary probe location; falls back to `parent` below
        let safe = |name: &str| -> String {
            name.chars()
                .map(|c| if matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
                .collect()
        };

        let mut found = None;
        // A present-but-unreadable cluster asset (wrong dtype/rank) is treated as ABSENT — the
        // cluster blend is optional, so a bad file must not abort the whole 翻唱 (matches the
        // original's missing-file skip). Warn and fall through to None.
        'dirs: for dir in [&model_dir, &parent] {
            let index_path = dir.join(format!("{}.index_vectors.npy", spk));
            if index_path.exists() {
                match app.inference.load_npy(&index_path) {
                    Ok(arr) => {
                        found = Some(sovits::ClusterAsset::FeatureIndex(
                            crate::inference::features::KnnIndex::new((*arr).clone()),
                        ));
                        break;
                    }
                    Err(e) => tracing::warn!("检索资产 {} 无法加载，跳过聚类混合：{}", index_path.display(), e),
                }
            }
            // kmeans 文件名用 speaker 名（config.speakers 反查 id）
            for (name, _) in entry.config.speakers.iter().filter(|(_, &id)| id == spk) {
                let kmeans_path = dir.join(format!("{}.centers.npy", safe(name)));
                if kmeans_path.exists() {
                    match app.inference.load_npy(&kmeans_path) {
                        Ok(arr) => {
                            found = Some(sovits::ClusterAsset::KmeansCenters(
                                crate::inference::features::KnnIndex::new((*arr).clone()),
                            ));
                            break 'dirs;
                        }
                        Err(e) => tracing::warn!("聚类资产 {} 无法加载，跳过聚类混合：{}", kmeans_path.display(), e),
                    }
                }
            }
        }
        found // None → pipeline logs the skip (mirrors the original's missing-file behavior)
    } else {
        None
    };

    let audio_buf =
        crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;

    let progress = progress_emitter(app_handle, node_id);
    tauri::async_runtime::spawn_blocking(move || {
        let model = sovits::SovitsModel {
            engine: &app.inference.engine,
            voice_session: &handle.session_id,
            contentvec_session: &cv_sid,
            rmvpe_session: &rmvpe_sid,
            mel_filters: mel.as_ref(),
            cluster: cluster.as_ref(),
            sample_rate: handle.sample_rate,
            hop_size,
            features_dim: dim,
            vol_embedding,
            unit_interpolate_mode,
            noise_channels: nch,
            min_frames: min_t,
        };
        sovits::run_pipeline(&model, &audio_buf, &options, &progress).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("推理任务失败: {}", e))?
}

// ─── detect_f0 (kept signature: audio path → f0 Hz @ 100 fps) ────────────────

#[tauri::command]
pub async fn detect_f0(
    state: State<'_, Arc<AppState>>,
    audio_path: String,
) -> Result<Vec<f32>, String> {
    let app = state.inner().clone();
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "音高检测模型")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "音高检测滤波器")?;
    let rmvpe_sid = app.inference.ensure_aux_loaded(&rmvpe_path).map_err(|e| e.to_string())?;
    let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;

    let audio_buf =
        crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;

    tauri::async_runtime::spawn_blocking(move || {
        let mono = crate::audio::resample::to_mono(&audio_buf);
        let wav16k = crate::inference::features::resample(
            &mono.samples,
            mono.sample_rate,
            crate::inference::f0::RMVPE_SR,
        );
        crate::inference::f0::rmvpe_detect(
            &app.inference.engine,
            &rmvpe_sid,
            &mel,
            &wav16k,
            crate::inference::f0::RVC_RMVPE_THRESHOLD,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("音高检测任务失败: {}", e))?
}

// ─── run_s2h (untouched) ─────────────────────────────────────────────────────

#[tauri::command]
pub async fn run_s2h(
    state: State<'_, Arc<AppState>>,
    phonemes: Vec<i64>,
    durations: Vec<i64>,
    pitches: Vec<f32>,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let s2h_model = state
        .models
        .list_by_type(&crate::models::ModelType::S2H)
        .first()
        .cloned()
        .ok_or_else(|| "No S2H model available".to_string())?;

    let session_id = state
        .inference
        .engine
        .load_model(&s2h_model.path)
        .map_err(|e| e.to_string())?;

    let score = crate::inference::s2h::ScoreInput {
        phonemes,
        durations,
        pitches,
    };

    let output = match crate::inference::s2h::infer(&state.inference.engine, &session_id, &score) {
        Ok(o) => {
            state.inference.engine.unload_model(&session_id);
            o
        }
        Err(e) => {
            state.inference.engine.unload_model(&session_id);
            return Err(e.to_string());
        }
    };

    let hubert: Vec<Vec<f32>> = output
        .hubert_features
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    let contentvec: Vec<Vec<f32>> = output
        .contentvec_features
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();

    Ok((hubert, contentvec))
}
