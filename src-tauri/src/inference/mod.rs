pub mod engine;
pub mod f0;
pub mod features;
pub mod nsf_hifigan;
pub mod rvc;
pub mod s2h;
pub mod sovits;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::Result;

/// Wire-contract options for run_rvc — mirrored 1:1 by src\lib\workflow\voiceDefaults.ts
/// (THE frontend source of truth). Struct-level serde default: any absent key falls back
/// to the Default impl below.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RvcOptions {
    pub f0_shift: f32,
    pub speaker_id: Option<u32>,
    pub index_ratio: f32,
    pub protect: f32,
    pub noise_scale: f32,
    pub rms_mix_rate: f32,
    pub l2_normalize: bool,
    pub resample_sr: u32,
    pub seed: u64,
}

impl Default for RvcOptions {
    fn default() -> Self {
        Self {
            f0_shift: 0.0,
            speaker_id: None,
            index_ratio: 0.75,
            protect: 0.33,
            noise_scale: 0.66666,
            rms_mix_rate: 0.25,
            l2_normalize: false,
            resample_sr: 0,
            seed: 0,
        }
    }
}

/// Wire-contract options for run_sovits (voiceDefaults.ts mirror).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SovitsOptions {
    pub f0_shift: f32,
    pub speaker_id: Option<u32>,
    pub noise_scale: f32,
    pub cluster_ratio: f32,
    pub loudness_envelope: f32,
    pub seed: u64,
}

impl Default for SovitsOptions {
    fn default() -> Self {
        Self {
            f0_shift: 0.0,
            speaker_id: None,
            noise_scale: 0.4,
            cluster_ratio: 0.0,
            loudness_envelope: 1.0,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesisResult {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Clone, Debug)]
pub enum VoiceBackendType {
    Rvc,
    SoVits,
}

struct LoadedVoice {
    _backend_type: VoiceBackendType,
    _model_path: PathBuf,
    session_id: String,
    sample_rate: u32,
    index: Option<Arc<rvc::RvcIndex>>,
}

/// Cheap cloneable view of a loaded voice for the pipelines.
pub struct VoiceHandle {
    pub session_id: String,
    pub sample_rate: u32,
    pub index: Option<Arc<rvc::RvcIndex>>,
}

pub struct InferenceManager {
    pub engine: engine::OnnxEngine,
    /// Voice sessions keyed by VOICE NAME — model delete/reimport calls unload_voice(name).
    loaded_voices: RwLock<HashMap<String, LoadedVoice>>,
    /// Aux model sessions (ContentVec variants, RMVPE) keyed by path — the generalization
    /// of the old single cached_f0_session slot.
    aux_sessions: RwLock<HashMap<PathBuf, String>>,
    /// Small .npy cache (RMVPE mel filters, so-vits cluster assets) keyed by path.
    npy_cache: RwLock<HashMap<PathBuf, Arc<ndarray::Array2<f32>>>>,
}

impl InferenceManager {
    pub fn new() -> Self {
        Self {
            engine: engine::OnnxEngine::new(),
            loaded_voices: RwLock::new(HashMap::new()),
            aux_sessions: RwLock::new(HashMap::new()),
            npy_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Load (or reuse) an aux ONNX session (ContentVec / RMVPE) cached by path.
    pub fn ensure_aux_loaded(&self, path: &Path) -> Result<String> {
        {
            let cached = self.aux_sessions.read();
            if let Some(sid) = cached.get(path) {
                if self.engine.is_loaded(sid) {
                    return Ok(sid.clone());
                }
            }
        }
        // Aux feature extractors (ContentVec / RMVPE) run on CPU: they're one-shot passes over the
        // WHOLE (up to full-song) signal, whose fp32 activations are the dominant VRAM consumer —
        // a 2-min song peaked ~9 GB with them on GPU. On CPU their RAM is unbounded-enough (32 GB)
        // and numerically they're MORE faithful than the TF32 GPU path (the E2E gate ran on CPU).
        // The voice synthesizer (the per-chunk hot loop) stays on the global device (GPU). Also
        // mem_pattern OFF — varying-length inputs would make the pattern planner over-reserve.
        let sid = self
            .engine
            .load_model_on(&path.to_path_buf(), false, engine::DeviceConfig::Cpu)?;
        self.aux_sessions
            .write()
            .insert(path.to_path_buf(), sid.clone());
        tracing::info!("Aux model cached: {}", path.display());
        Ok(sid)
    }

    /// Load a .npy as Array2<f32>, cached by path (mel filters / cluster assets).
    pub fn load_npy(&self, path: &Path) -> Result<Arc<ndarray::Array2<f32>>> {
        if let Some(arr) = self.npy_cache.read().get(path) {
            return Ok(arr.clone());
        }
        let arr: ndarray::Array2<f32> = ndarray_npy::read_npy(path).map_err(|e| {
            crate::UtaiError::Model(format!("加载 npy 失败 '{}': {}", path.display(), e))
        })?;
        let arr = Arc::new(arr);
        self.npy_cache
            .write()
            .insert(path.to_path_buf(), arr.clone());
        Ok(arr)
    }

    pub fn is_voice_loaded(&self, name: &str) -> bool {
        let voices = self.loaded_voices.read();
        if let Some(voice) = voices.get(name) {
            self.engine.is_loaded(&voice.session_id)
        } else {
            false
        }
    }

    pub fn load_voice(
        &self,
        name: &str,
        model_path: &PathBuf,
        backend_type: VoiceBackendType,
        sample_rate: u32,
        index_path: Option<&PathBuf>,
    ) -> Result<()> {
        if self.is_voice_loaded(name) {
            return Ok(());
        }
        self.unload_voice(name);
        // mem_pattern OFF: voice models run per-chunk with varying T (RVC) / whole-segment T
        // (SoVITS) — dynamic shapes where ORT's memory pattern over-reserves VRAM.
        let session_id = self.engine.load_model_with(model_path, false)?;

        let index = match (&backend_type, index_path) {
            (VoiceBackendType::Rvc, Some(path)) if path.exists() => {
                match rvc::RvcIndex::load(path) {
                    Ok(idx) => Some(Arc::new(idx)),
                    Err(e) => {
                        tracing::warn!("Failed to load index, continuing without: {}", e);
                        None
                    }
                }
            }
            _ => None,
        };

        let voice = LoadedVoice {
            _backend_type: backend_type,
            _model_path: model_path.clone(),
            session_id,
            sample_rate,
            index,
        };
        self.loaded_voices.write().insert(name.to_string(), voice);
        Ok(())
    }

    pub fn voice_handle(&self, name: &str) -> Result<VoiceHandle> {
        let voices = self.loaded_voices.read();
        let voice = voices.get(name).ok_or_else(|| {
            crate::UtaiError::Inference(format!("模型 '{}' 尚未加载", name))
        })?;
        Ok(VoiceHandle {
            session_id: voice.session_id.clone(),
            sample_rate: voice.sample_rate,
            index: voice.index.clone(),
        })
    }

    pub fn unload_voice(&self, name: &str) {
        let mut voices = self.loaded_voices.write();
        if let Some(voice) = voices.remove(name) {
            self.engine.unload_model(&voice.session_id);
        }
        // Model files may be replaced on reimport — drop cached npy assets (cheap reloads)
        // so a stale cluster index / retrieval asset can't outlive its file.
        self.npy_cache.write().clear();
    }
}
