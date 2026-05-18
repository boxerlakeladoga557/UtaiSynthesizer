use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::audio::effects::Effect;
use crate::AppState;

#[derive(serde::Serialize)]
pub struct AudioFileInfo {
    pub duration_ms: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub peaks: Vec<f32>,
}

#[derive(serde::Deserialize)]
pub struct EffectRequest {
    pub audio_path: String,
    pub effects: Vec<Effect>,
    pub output_path: Option<String>,
}

#[derive(serde::Serialize)]
pub struct EffectResult {
    pub output_path: String,
    pub duration_secs: f64,
}

#[tauri::command]
pub async fn load_audio_file(path: String) -> Result<AudioFileInfo, String> {
    let buf = crate::audio::load_wav(&PathBuf::from(&path)).map_err(|e| e.to_string())?;

    let duration_ms = buf.duration_secs() * 1000.0;
    let peaks = extract_peaks(&buf.samples, buf.channels, 4000);

    Ok(AudioFileInfo {
        duration_ms,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        peaks,
    })
}

fn extract_peaks(samples: &[f32], channels: u16, target_count: usize) -> Vec<f32> {
    let frame_count = samples.len() / channels as usize;
    if frame_count == 0 {
        return vec![];
    }

    let count = target_count.min(frame_count);
    let frames_per_peak = frame_count as f64 / count as f64;
    let mut peaks = Vec::with_capacity(count);

    for i in 0..count {
        let start = (i as f64 * frames_per_peak) as usize;
        let end = (((i + 1) as f64) * frames_per_peak) as usize;
        let end = end.min(frame_count);

        let mut max_val = 0.0f32;
        for frame in start..end {
            let idx = frame * channels as usize;
            if idx < samples.len() {
                max_val = max_val.max(samples[idx].abs());
            }
        }
        peaks.push(max_val);
    }

    peaks
}

#[tauri::command]
pub async fn process_effects(
    state: State<'_, Arc<AppState>>,
    request: EffectRequest,
) -> Result<EffectResult, String> {
    let mut buffer = crate::audio::load_wav(&PathBuf::from(&request.audio_path))
        .map_err(|e| e.to_string())?;

    let nsf_session: Option<&str> = None;

    for effect in &request.effects {
        buffer = crate::audio::effects::apply_effect(
            &buffer,
            effect,
            &state.inference.engine,
            nsf_session,
        )
        .map_err(|e| e.to_string())?;
    }

    let output_path = request.output_path.unwrap_or_else(|| {
        let input = PathBuf::from(&request.audio_path);
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        input
            .with_file_name(format!("{}_processed.wav", stem))
            .to_string_lossy()
            .to_string()
    });

    crate::audio::save_wav(&PathBuf::from(&output_path), &buffer)
        .map_err(|e| e.to_string())?;

    Ok(EffectResult {
        output_path,
        duration_secs: buffer.duration_secs(),
    })
}

#[tauri::command]
pub async fn export_audio(
    state: State<'_, Arc<AppState>>,
    _output_path: String,
    _format: String,
    _sample_rate: u32,
    _normalize: bool,
) -> Result<(), String> {
    let proj = state.project.read();
    let _project = proj
        .as_ref()
        .ok_or_else(|| "No project open".to_string())?;

    Err("Export requires rendered tracks — pending full pipeline integration".to_string())
}
