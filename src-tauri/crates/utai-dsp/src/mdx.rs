//! Legacy MDX-Net (UVR SeperateMDX) helpers. The chunk loop lives in the app's
//! pipeline (it drives ORT); these are the pure-DSP pieces of its recipe:
//! trim-padded mixture, numpy-hanning OLA window, and the "zero the 3 lowest
//! freq bins" input tweak both reference implementations apply.
//!
//! STFT/iSTFT for MDX-Net use the TORCH conventions (center=True reflect pad,
//! periodic hann) — i.e. the existing `stft::StftProcessor`, NOT vr.rs's
//! librosa functions.

/// numpy.hanning(n) — SYMMETRIC hann (unlike the periodic STFT window):
/// w[i] = 0.5 − 0.5·cos(2πi/(n−1)); n==1 → [1.0].
pub fn np_hanning(n: usize) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

/// UVR MDX demix padding: [zeros(trim), mix, zeros(gen_size + trim − L % gen_size)].
/// Returns the padded stereo pair. Padding constants: trim = n_fft/2,
/// gen_size = chunk_size − 2·trim.
pub fn mdx_pad_mix(
    left: &[f32],
    right: &[f32],
    trim: usize,
    gen_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = left.len();
    let tail = gen_size + trim - (n % gen_size); // reference formula (L%gen==0 → gen+trim)
    let total = trim + n + tail;
    let mut l = vec![0.0f32; total];
    let mut r = vec![0.0f32; total];
    l[trim..trim + n].copy_from_slice(left);
    r[trim..trim + n].copy_from_slice(right);
    (l, r)
}

/// Zero the 3 lowest freq rows across all 4 CaC planes of a flat
/// [4, dim_f, dim_t] input (`spek[:, :, :3, :] *= 0` in both references).
pub fn zero_low_bins(cac: &mut [f32], dim_f: usize, dim_t: usize) {
    for plane in 0..4 {
        for row in 0..3.min(dim_f) {
            let base = (plane * dim_f + row) * dim_t;
            for v in &mut cac[base..base + dim_t] {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hanning_symmetric_endpoints() {
        let w = np_hanning(5);
        assert!((w[0]).abs() < 1e-7 && (w[4]).abs() < 1e-7);
        assert!((w[2] - 1.0).abs() < 1e-7); // symmetric peak at center
        assert_eq!(np_hanning(1), vec![1.0]);
    }

    #[test]
    fn pad_math_matches_reference() {
        // KARA geometry: n_fft 6144 → trim 3072, chunk 261120 → gen 254976
        let trim = 3072;
        let gen = 261120 - 2 * trim;
        let n = 1_000_000;
        let (l, _r) = mdx_pad_mix(&vec![0.5; n], &vec![0.5; n], trim, gen);
        let expected_tail = gen + trim - (n % gen);
        assert_eq!(l.len(), trim + n + expected_tail);
        assert_eq!(l[trim - 1], 0.0);
        assert_eq!(l[trim], 0.5);
    }
}
