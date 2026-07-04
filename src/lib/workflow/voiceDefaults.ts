/**
 * THE voice-node parameter contract — the single source of truth shared by the node UIs
 * (RvcNode / SoVitsNode) and the workflow engine, and the exact shape the Rust pipeline's
 * options deserialization must mirror (run_rvc / run_sovits in src-tauri).
 *
 * The engine serializes EXACTLY these snake_case keys as the `options` object of the invoke
 * payload `{ voiceName, modelPath, audioPath, options }` — nothing else (the old
 * `shallowDiffusion` arg is gone: shallow-diffusion / auto-f0 / spk-mix are DEFERRED by user
 * decision, so they have no params and no UI).
 *
 * Node params store the SAME snake_case keys (plus `voiceName` / `modelPath`), so there is no
 * UI-key → wire-key mapping layer to drift: an absent key means "use the default below".
 * f0 method is rmvpe-only for now — no selector param.
 */

export interface RvcOptions {
  /** Pitch shift in semitones, -24..24. */
  f0_shift: number;
  /** Target speaker index for multi-speaker models; null = 0 (single-speaker default). */
  speaker_id: number | null;
  /** KNN index feature blend, 0..1. */
  index_ratio: number;
  /** Voiceless-consonant/breath protection, 0..0.5 — 0.5 means OFF. */
  protect: number;
  /** Synthesis randomness, 0..1. */
  noise_scale: number;
  /** Output-loudness envelope mix vs the input's, 0..1. */
  rms_mix_rate: number;
  /** L2-normalize ContentVec features before the index lookup (official pipeline does NOT). */
  l2_normalize: boolean;
  /** Output resample rate; 0 = keep the model's sample rate. */
  resample_sr: number;
  seed: number;
}

export interface SovitsOptions {
  /** Pitch shift in semitones. */
  f0_shift: number;
  /** Target speaker index for multi-speaker models; null = 0. */
  speaker_id: number | null;
  /** Synthesis randomness, 0..1. */
  noise_scale: number;
  /** Cluster-model / feature-index blend, 0..1; 0 = off. */
  cluster_ratio: number;
  /** Input-loudness-envelope replacement mix, 0..1 — 1.0 means OFF (keep output loudness). */
  loudness_envelope: number;
  seed: number;
}

export const RVC_DEFAULTS: RvcOptions = {
  f0_shift: 0,
  speaker_id: null,
  index_ratio: 0.75,
  protect: 0.33,
  noise_scale: 0.66666,
  rms_mix_rate: 0.25,
  l2_normalize: false,
  resample_sr: 0,
  seed: 0,
};

export const SOVITS_DEFAULTS: SovitsOptions = {
  f0_shift: 0,
  speaker_id: null,
  noise_scale: 0.4,
  cluster_ratio: 0,
  loudness_envelope: 1.0,
  seed: 0,
};

/**
 * The params object persisted on an rvc/sovits WorkflowNode (`WorkflowNode.params` in
 * types/project.ts — untyped Record there; this documents the shape):
 *   - `voiceName`  — registry name of the model (list_models entry `.name`), invoke `voiceName`
 *   - `modelPath`  — the entry's `.path`, invoke `modelPath`
 *   - plus any subset of RvcOptions / SovitsOptions keys VERBATIM (absent = default above).
 */
export interface VoiceNodeParams extends Partial<RvcOptions & SovitsOptions> {
  voiceName?: string;
  modelPath?: string;
}

/**
 * Build the wire `options` object: contract defaults overlaid with any contract keys the node
 * params carry. ONLY keys present in `defaults` are emitted — node-side extras (voiceName,
 * modelPath, ...) never leak into the options payload.
 */
export function buildVoiceOptions<T extends object>(
  defaults: T,
  params: Record<string, unknown>,
): T {
  const out = { ...defaults } as Record<string, unknown>;
  for (const key of Object.keys(defaults)) {
    if (params[key] !== undefined) out[key] = params[key];
  }
  return out as T;
}
