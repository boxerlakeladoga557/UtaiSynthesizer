//! formant.rs — pure-Rust formant (spectral-envelope) shifter.
//!
//! Warps the spectral ENVELOPE (the slowly-varying formant structure, isolated via low-quefrency
//! cepstral liftering) while leaving the fine structure (pitch harmonics) in place — so the PITCH is
//! unchanged and only the timbre / "gender" moves. `ratio > 1` raises the formants (brighter /
//! "younger"), `< 1` lowers them. This is the WORLD-free path (WORLD was never actually integrated in
//! this repo): built on the existing rustfft `StftProcessor`, shared by the ② vocal render (a per-frame
//! ratio sampled from the formant lane) and the audio-track cover nodes (a single scalar ratio).
//!
//! Ear-validated, not bit-exact (like the rest of the singing path): the log-magnitude round-trip
//! (ln → cepstrum → lifter → exp) introduces tiny numeric error, so a frame whose ratio is ≈ 1 is
//! passed through verbatim (original complex bins) — a flat/neutral lane setting is near-lossless.

use crate::stft::{StftConfig, StftProcessor};
use ndarray::Array3;
use rustfft::{num_complex::Complex, FftPlanner};

const N_FFT: usize = 2048;
const HOP: usize = 512;
/// Cepstral lifter cutoff (quefrency bins kept, each end). Must sit BELOW the pitch period so the
/// envelope smooths OVER the harmonic spacing without swallowing the pitch peak: 48 samples ≈ 1.09 ms
/// ≈ a 918 Hz cutoff at 44.1 k, so f0 up to ~900 Hz stays fine-structure (pitch-safe for singing).
/// Larger = smoother envelope / fewer formant details; smaller = coarser. Ear-tunable.
const LIFTER_Q: usize = 48;

/// Warp the formant envelope of a mono signal. `ratio_at(frame_center_sample)` returns the warp ratio
/// for the STFT frame whose (unpadded) center lands at that sample index — return a constant for a
/// scalar shift, or sample an envelope for a per-frame lane. Output length == input length. Pitch is
/// preserved (only the envelope moves). Frames with ratio ≈ 1 pass through unchanged.
pub fn formant_warp<F: Fn(usize) -> f32>(mono: &[f32], ratio_at: F) -> Vec<f32> {
    if mono.len() < N_FFT {
        return mono.to_vec(); // too short to STFT meaningfully — pass through
    }
    let proc = StftProcessor::new(StftConfig { n_fft: N_FFT, hop_length: HOP, win_length: N_FFT });
    let spec = proc.stft(mono); // [freq_bins, frames, 2]
    let freq_bins = proc.freq_bins();
    let n_frames = spec.shape()[1];

    // Cepstrum transforms (size N_FFT). rustfft is UNNORMALIZED both ways, so fft_fwd∘fft_inv = N·id;
    // dividing the cepstrum by N once (the `norm` below) makes the pair recover the log-magnitude.
    let mut planner = FftPlanner::<f32>::new();
    let fft_fwd = planner.plan_fft_forward(N_FFT);
    let fft_inv = planner.plan_fft_inverse(N_FFT);
    let norm = 1.0 / N_FFT as f32;

    let mut out = Array3::<f32>::zeros((freq_bins, n_frames, 2));

    let mut logmag_full = vec![0.0f32; N_FFT];
    let mut cep = vec![Complex::new(0.0f32, 0.0f32); N_FFT];
    let mut env = vec![Complex::new(0.0f32, 0.0f32); N_FFT];

    for t in 0..n_frames {
        // Frame t's center in the ORIGINAL (unpadded) signal is t*HOP (StftProcessor pads by N_FFT/2
        // up front and the window is centered, so center = t*HOP + N_FFT/2 − pad = t*HOP).
        let ratio = ratio_at(t * HOP);
        if (ratio - 1.0).abs() < 1e-3 {
            for k in 0..freq_bins {
                out[[k, t, 0]] = spec[[k, t, 0]];
                out[[k, t, 1]] = spec[[k, t, 1]];
            }
            continue;
        }

        // 1. full-spectrum log-magnitude (mirror the positive-freq bins into the upper half so the
        //    cepstrum of this real, even sequence stays real & even).
        for k in 0..freq_bins {
            let re = spec[[k, t, 0]];
            let im = spec[[k, t, 1]];
            logmag_full[k] = ((re * re + im * im).sqrt() + 1e-9).ln();
        }
        for k in freq_bins..N_FFT {
            logmag_full[k] = logmag_full[N_FFT - k];
        }

        // 2. cepstrum = IFFT(log-magnitude)
        for k in 0..N_FFT {
            cep[k] = Complex::new(logmag_full[k], 0.0);
        }
        fft_inv.process(&mut cep);

        // 3. lifter: keep the low quefrency (both symmetric ends) = the smooth spectral envelope.
        for k in 0..N_FFT {
            let keep = k < LIFTER_Q || k >= N_FFT - LIFTER_Q + 1;
            let w = if keep { norm } else { 0.0 };
            env[k] = Complex::new(cep[k].re * w, cep[k].im * w);
        }

        // 4. envelope log-magnitude = FFT(liftered cepstrum). env[k].re holds it (imag ≈ 0).
        fft_fwd.process(&mut env);

        // 5. warp the envelope by `ratio`, keep the fine structure, rebuild the complex spectrum
        //    (fine = logmag − env is the harmonic detail = pitch; only the envelope is remapped).
        for k in 0..freq_bins {
            let fine = logmag_full[k] - env[k].re;
            let env_w = lerp_env(&env, k as f32 / ratio, freq_bins);
            let new_mag = (fine + env_w).exp();
            let re = spec[[k, t, 0]];
            let im = spec[[k, t, 1]];
            let phase = im.atan2(re);
            out[[k, t, 0]] = new_mag * phase.cos();
            out[[k, t, 1]] = new_mag * phase.sin();
        }
    }

    proc.istft(&out, mono.len())
}

/// Linear-interpolate the envelope log-magnitude (env[k].re) at a fractional bin `src`, holding flat
/// past both ends (bin 0 and the Nyquist bin freq_bins−1).
fn lerp_env(env: &[Complex<f32>], src: f32, freq_bins: usize) -> f32 {
    if src <= 0.0 {
        return env[0].re;
    }
    let hi = freq_bins - 1;
    if src >= hi as f32 {
        return env[hi].re;
    }
    let lo = src.floor() as usize;
    let frac = src - lo as f32;
    env[lo].re * (1.0 - frac) + env[lo + 1].re * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * PI * freq * i as f32 / sr).sin()).collect()
    }

    /// A harmonic-rich tone (sawtooth-ish) at f0 with a synthetic formant bump — the realistic input.
    fn harmonic(f0: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let mut s = 0.0f32;
                for h in 1..=10 {
                    // a bump around the 4th harmonic makes an envelope for the warp to actually move
                    let amp = 1.0 / h as f32 * (1.0 + 2.0 * (-((h as f32 - 4.0).powi(2)) / 4.0).exp());
                    s += amp * (2.0 * PI * f0 * h as f32 * t).sin();
                }
                s * 0.1
            })
            .collect()
    }

    /// Autocorrelation-based fundamental period (samples) — used to check pitch is preserved.
    fn est_period(x: &[f32], min_lag: usize, max_lag: usize) -> usize {
        let mut best = min_lag;
        let mut best_val = f32::MIN;
        for lag in min_lag..=max_lag.min(x.len() - 1) {
            let mut acc = 0.0f32;
            for i in 0..x.len() - lag {
                acc += x[i] * x[i + lag];
            }
            if acc > best_val {
                best_val = acc;
                best = lag;
            }
        }
        best
    }

    #[test]
    fn ratio_one_is_near_lossless() {
        let sr = 44100.0;
        let x = sine(440.0, sr, 8192);
        let y = formant_warp(&x, |_| 1.0);
        assert_eq!(y.len(), x.len());
        // interior (skip the first/last window where COLA is partial)
        let lo = N_FFT;
        let hi = x.len() - N_FFT;
        let err: f32 = (lo..hi).map(|i| (x[i] - y[i]).abs()).fold(0.0, f32::max);
        assert!(err < 1e-3, "ratio=1 should be near-lossless, max interior err = {err}");
    }

    #[test]
    fn preserves_pitch() {
        let sr = 44100.0;
        let f0 = 200.0;
        let x = harmonic(f0, sr, 16384);
        let expected = (sr / f0).round() as usize; // ~220 samples
        for ratio in [0.7f32, 1.3, 1.6] {
            let y = formant_warp(&x, |_| ratio);
            assert_eq!(y.len(), x.len());
            assert!(y.iter().all(|v| v.is_finite()), "ratio={ratio} produced non-finite samples");
            let p = est_period(&y[N_FFT..y.len() - N_FFT], expected - 30, expected + 30);
            assert!(
                (p as i32 - expected as i32).abs() <= 6,
                "formant warp ratio={ratio} moved the pitch: period {p} vs expected {expected}"
            );
        }
    }

    #[test]
    fn moves_the_envelope() {
        // Raising the formants (ratio>1) should shift spectral energy upward — sanity that it does
        // SOMETHING (not a no-op) while staying bounded.
        let sr = 44100.0;
        let x = harmonic(200.0, sr, 16384);
        let y = formant_warp(&x, |_| 1.5);
        let ex: f32 = x.iter().map(|v| v * v).sum();
        let ey: f32 = y.iter().map(|v| v * v).sum();
        assert!(ey > 0.0 && ey.is_finite());
        // energy stays within a sane band (no runaway from the exp())
        assert!(ey < ex * 10.0 && ey > ex * 0.1, "energy blew up: in={ex} out={ey}");
        // and the output actually differs from the input
        let diff: f32 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).sum::<f32>() / x.len() as f32;
        assert!(diff > 1e-4, "ratio=1.5 barely changed the signal (diff={diff})");
    }
}
