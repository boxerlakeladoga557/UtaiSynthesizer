use std::path::{Path, PathBuf};

/// Resolve the Python interpreter for a bundled sidecar/tool. Priority:
///   1. the tool's own venv:        `<venv_dir>/.venv/Scripts/python.exe`
///   2. the bundled portable Python: `<app_dir>/python/python.exe`
///   3. the system `python` on PATH (dev fallback)
///
/// `venv_dir` is the directory CONTAINING `.venv` (e.g. `app_dir/converter`), NOT the `.venv` itself.
/// Single source of truth: this was previously copy-pasted as `find_converter_python` in both
/// models/convert.rs and commands/msst_models.rs (and a broken cwd-relative, venv-skipping variant in
/// training/mod.rs). Windows-only layout (`Scripts/python.exe`), matching the original call sites.
pub fn find_python(venv_dir: &Path, app_dir: &Path) -> PathBuf {
    let venv = venv_dir.join(".venv").join("Scripts").join("python.exe");
    if venv.exists() {
        return venv;
    }
    let embedded = app_dir.join("python").join("python.exe");
    if embedded.exists() {
        return embedded;
    }
    PathBuf::from("python")
}

/// Windows `CREATE_NO_WINDOW` process-creation flag — pass to `Command::creation_flags(...)` so spawned
/// console tools (ffmpeg, powershell) don't flash a black console window. Was the bare magic
/// `0x08000000` repeated at every spawn site.
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Build a `std::process::Command` for a bundled Python tool with the shared spawn hygiene applied.
/// Single source of truth for EVERY python spawn (converter, index extractor, training):
///   - `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`: a PIPED stdout/stderr on Windows defaults to the
///     ANSI codepage, so one CJK character in a `print()` raises UnicodeEncodeError AFTER the tool
///     already wrote its artifacts — the spawn "fails" with the files on disk (phantom import).
///   - `CREATE_NO_WINDOW`: no console flash.
/// Async call sites convert with `tokio::process::Command::from(python_command(...))` — the flags
/// and envs carry over.
pub fn python_command(python: &Path) -> std::process::Command {
    let mut cmd = std::process::Command::new(python);
    cmd.env("PYTHONIOENCODING", "utf-8");
    cmd.env("PYTHONUTF8", "1");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Extract every `.dll` entry whose archive path satisfies `matches` from `zip_path` into `dest_dir`
/// (flattened to its basename). Single source for the CUDA-runtime downloader's nupkg + wheel DLL
/// extraction — previously `extract_nupkg_dlls` / `extract_wheel_dlls`, byte-identical except for
/// `starts_with` vs `contains`, now expressed by the caller's `matches` closure.
pub fn extract_zip_dlls(zip_path: &Path, dest_dir: &Path, matches: impl Fn(&str) -> bool) -> crate::Result<()> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| crate::UtaiError::Audio(format!("Zip open: {}", e)))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| crate::UtaiError::Audio(format!("Zip entry: {}", e)))?;
        let name = entry.name().to_string();
        if name.ends_with(".dll") && matches(&name) {
            let filename = name.rsplit('/').next().unwrap_or(&name);
            let dest = dest_dir.join(filename);
            let mut out = std::fs::File::create(&dest)?;
            std::io::copy(&mut entry, &mut out)?;
            tracing::info!("Extracted: {}", dest.display());
        }
    }
    Ok(())
}
