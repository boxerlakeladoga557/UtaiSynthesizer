//! UVR VR-arch DSP — multiband analysis/synthesis, mask post-processing, and a
//! scipy-exact polyphase resampler.
//!
//! Reference semantics: UVR / audio-separator `spec_utils.py` + `vr_separator.py`
//! (local authoritative copy: D:\MyDev\ARCHIVE\MSSTRVCv2\MSST\modules\vocal_remover).
//! Deliberate reference deviations, all SNR-gated at E2E (converter/verify/README.md):
//! - STFT/iSTFT here follow LIBROSA conventions (center pad with ZEROS, NOLA
//!   window-sum normalization) — NOT the torch conventions of `stft.rs` (reflect
//!   pad). Keep the two lineages separate; do not merge them.
//! - Internally computed in f64 (librosa's float64-window STFT path / its float64
//!   synthesis chain), complex values stored as f32 pairs (complex64 equivalence).
//! - Synthesis inter-band UPSAMPLING uses the same kaiser-β5 polyphase design as
//!   the analysis decimation (scipy `resample_poly`) instead of libsamplerate
//!   "sinc_fastest" — this matches what UVR itself runs on macOS-ARM (`polyphase`
//!   everywhere) and is a sanctioned original variant.
//! - `cmb_spectrogram_to_wave`'s per-band scratch spectrum is ZEROED (audio-separator
//!   master behavior); UVR allocates it uninitialized (np.ndarray) — a bug, not a
//!   semantic to copy.

use ndarray::Array3;
use rustfft::{num_complex::Complex, FftPlanner};

// ─── Band / model params (mirrors the converter's uvr_vr JSON `bands` table) ─

#[derive(Debug, Clone)]
pub struct VrBandParam {
    pub sr: u32,
    pub hl: usize,
    pub n_fft: usize,
    pub crop_start: usize,
    pub crop_stop: usize,
    /// Filters: negative or missing (encoded as i64::MIN sentinel? no — Option) per band.
    pub hpf_start: Option<i64>,
    pub hpf_stop: Option<i64>,
    pub lpf_start: Option<i64>,
    pub lpf_stop: Option<i64>,
    /// v5.1 per-band spectrum-domain channel transform ("mid_side"|"mid_side_c"|"stereo_n").
    pub convert_channels: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VrParams {
    pub bins: usize,
    pub pre_filter_start: i64,
    pub pre_filter_stop: i64,
    pub is_v51: bool,
    /// v5.0 GLOBAL waveform-domain transforms (mutually exclusive in practice).
    pub reverse: bool,
    pub mid_side: bool,
    pub mid_side_b2: bool,
    pub bands: Vec<VrBandParam>,
    pub window_size: usize,
    pub offset: usize,
    pub aggr_split_bin: usize,
    pub primary_non_accom: bool,
}

/// Stereo complex spectrogram: per channel [bins, frames, 2] (re, im) f32 —
/// the complex64 equivalent of the reference chain.
pub struct VrSpec {
    pub l: Array3<f32>,
    pub r: Array3<f32>,
}

// ─── scipy.signal.resample_poly (window=('kaiser', 5.0)) — exact replication ─

fn bessel_i0(x: f64) -> f64 {
    // Series Σ ((x/2)^2k / (k!)^2) — converges fast for the β≈5 range used here.
    let t = x * x / 4.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut k = 1.0f64;
    loop {
        term *= t / (k * k);
        sum += term;
        if term < sum * 1e-16 {
            return sum;
        }
        k += 1.0;
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta)) — symmetric
/// kaiser-windowed sinc lowpass, scale=True (DC gain normalized to 1).
fn firwin_kaiser(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = 0.5 * (numtaps - 1) as f64;
    let i0b = bessel_i0(beta);
    let mut h: Vec<f64> = (0..numtaps)
        .map(|i| {
            let m = i as f64 - alpha;
            let r = if alpha > 0.0 { m / alpha } else { 0.0 };
            let win = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            cutoff * sinc(cutoff * m) * win
        })
        .collect();
    let s: f64 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= s;
    }
    h
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// upfirdn output length (scipy `_output_len`).
fn upfirdn_out_len(len_h: usize, n_in: usize, up: usize, down: usize) -> usize {
    ((n_in - 1) * up + len_h - 1) / down + 1
}

/// scipy.signal.resample_poly(x, up, down, window=('kaiser', 5.0), padtype='constant').
/// This is the res_type="polyphase" path of librosa.resample — the ONLY resampler the
/// VR analysis cascade actually hits for the shipped multiband configs (and UVR's
/// macOS-ARM synthesis resampler). f64 arithmetic throughout.
pub fn resample_poly(x: &[f64], up_in: usize, down_in: usize) -> Vec<f64> {
    let g = gcd(up_in, down_in).max(1);
    let up = up_in / g;
    let down = down_in / g;
    if up == 1 && down == 1 {
        return x.to_vec();
    }
    let n_in = x.len();
    if n_in == 0 {
        return vec![];
    }
    let n_out = (n_in * up) / down + usize::from((n_in * up) % down != 0);
    let max_rate = up.max(down);
    let f_c = 1.0 / max_rate as f64;
    let half_len = 10 * max_rate;
    let mut h = firwin_kaiser(2 * half_len + 1, f_c, 5.0);
    for v in h.iter_mut() {
        *v *= up as f64;
    }
    let n_pre_pad = down - half_len % down; // scipy: may equal `down` when divisible
    let n_pre_remove = (half_len + n_pre_pad) / down;
    let mut n_post_pad = 0usize;
    while upfirdn_out_len(h.len() + n_pre_pad + n_post_pad, n_in, up, down)
        < n_out + n_pre_remove
    {
        n_post_pad += 1;
    }
    let mut hp = vec![0.0f64; n_pre_pad];
    hp.extend_from_slice(&h);
    hp.extend(std::iter::repeat(0.0).take(n_post_pad));

    let total = upfirdn_out_len(hp.len(), n_in, up, down);
    let lh = hp.len() as i64;
    let upi = up as i64;
    let mut y = vec![0.0f64; total];
    for (n, out) in y.iter_mut().enumerate() {
        let t = n as i64 * down as i64;
        let kmin_raw = t - lh + 1;
        let kmin = if kmin_raw <= 0 { 0 } else { (kmin_raw + upi - 1) / upi };
        let kmax = (t / upi).min(n_in as i64 - 1);
        let mut acc = 0.0f64;
        let mut k = kmin;
        while k <= kmax {
            acc += x[k as usize] * hp[(t - k * upi) as usize];
            k += 1;
        }
        *out = acc;
    }
    y[n_pre_remove..n_pre_remove + n_out].to_vec()
}

/// f32 convenience wrapper (analysis cascade waves are float32 in the reference).
pub fn resample_poly_f32(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    resample_poly(&x64, up, down).into_iter().map(|v| v as f32).collect()
}

// ─── librosa-convention STFT / iSTFT ─────────────────────────────

fn hann_periodic_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// librosa.stft(y, n_fft, hop_length): hann periodic (win_length = n_fft),
/// center=True with ZERO padding (librosa ≥0.10 default pad_mode='constant' —
/// the behavior of our reference environment), f64 compute, f32 storage.
/// Output [n_fft/2+1, frames, 2], frames = 1 + len/hop.
pub fn librosa_stft(signal: &[f32], n_fft: usize, hop: usize) -> Array3<f32> {
    let freq_bins = n_fft / 2 + 1;
    let pad = n_fft / 2;
    let padded_len = signal.len() + 2 * pad;
    let num_frames = if padded_len >= n_fft { (padded_len - n_fft) / hop + 1 } else { 0 };

    let window = hann_periodic_f64(n_fft);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut result = Array3::<f32>::zeros((freq_bins, num_frames, 2));
    let mut buffer = vec![Complex::new(0.0f64, 0.0f64); n_fft];
    for frame in 0..num_frames {
        let start = frame * hop; // position in the zero-padded signal
        for (i, b) in buffer.iter_mut().enumerate() {
            let pos = start + i;
            // zero-padded sample lookup: [0, pad) and [pad+len, ..) are zeros
            let v = if pos >= pad && pos < pad + signal.len() {
                signal[pos - pad] as f64
            } else {
                0.0
            };
            *b = Complex::new(v * window[i], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..freq_bins {
            result[[bin, frame, 0]] = buffer[bin].re as f32;
            result[[bin, frame, 1]] = buffer[bin].im as f32;
        }
    }
    result
}

/// librosa.istft(spec, hop_length, length=None): hann periodic, center=True →
/// output hop*(frames-1) samples, NOLA window-sum normalization. f64 output
/// (the reference synthesis chain runs float64 from here on).
pub fn librosa_istft_f64(spec: &Array3<f32>, n_fft: usize, hop: usize) -> Vec<f64> {
    let freq_bins = n_fft / 2 + 1;
    debug_assert_eq!(spec.shape()[0], freq_bins);
    let num_frames = spec.shape()[1];
    if num_frames == 0 {
        return vec![];
    }
    let pad = n_fft / 2;
    let full_len = (num_frames - 1) * hop + n_fft;
    let out_len = hop * (num_frames - 1);

    let window = hann_periodic_f64(n_fft);
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut output = vec![0.0f64; full_len];
    let mut window_sum = vec![0.0f64; full_len];
    let mut buffer = vec![Complex::new(0.0f64, 0.0f64); n_fft];
    let norm = 1.0 / n_fft as f64;
    for frame in 0..num_frames {
        for bin in 0..freq_bins {
            buffer[bin] = Complex::new(
                spec[[bin, frame, 0]] as f64,
                spec[[bin, frame, 1]] as f64,
            );
        }
        for bin in freq_bins..n_fft {
            let mirror = n_fft - bin;
            buffer[bin] = Complex::new(buffer[mirror].re, -buffer[mirror].im);
        }
        ifft.process(&mut buffer);
        let start = frame * hop;
        for i in 0..n_fft {
            output[start + i] += buffer[i].re * norm * window[i];
            window_sum[start + i] += window[i] * window[i];
        }
    }
    let tiny = 1e-15f64;
    for i in 0..full_len {
        if window_sum[i] > tiny {
            output[i] /= window_sum[i];
        }
    }
    output[pad..pad + out_len].to_vec()
}

// ─── Channel transforms ──────────────────────────────────────────

/// v5.0 GLOBAL waveform-domain transform, applied to each band's resampled wave
/// BEFORE its STFT (spec_utils.wave_to_spectrogram, not-v51 branch).
pub fn v4_wave_transform(params: &VrParams, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    if params.reverse {
        let l: Vec<f32> = left.iter().rev().copied().collect();
        let r: Vec<f32> = right.iter().rev().copied().collect();
        (l, r)
    } else if params.mid_side {
        let l: Vec<f32> = left.iter().zip(right).map(|(a, b)| (a + b) / 2.0).collect();
        let r: Vec<f32> = left.iter().zip(right).map(|(a, b)| a - b).collect();
        (l, r)
    } else if params.mid_side_b2 {
        let l: Vec<f32> = left.iter().zip(right).map(|(a, b)| b + 0.5 * a).collect();
        let r: Vec<f32> = left.iter().zip(right).map(|(a, b)| a - 0.5 * b).collect();
        (l, r)
    } else {
        (left.to_vec(), right.to_vec())
    }
}

/// v5.1 per-band SPECTRUM-domain transform (spec_utils.convert_channels), applied
/// after the band's STFT. No-op when the band has no convert_channels.
pub fn v51_convert_channels(spec: &mut VrSpec, cc: Option<&str>) {
    let (a0, a1, b0, b1) = match cc {
        Some("mid_side_c") => (1.0f32, 0.25f32, -0.25f32, 1.0f32), // L' = L + 0.25R ; R' = R - 0.25L
        Some("mid_side") => (0.5, 0.5, 1.0, -1.0),                 // L' = (L+R)/2   ; R' = L - R
        Some("stereo_n") => (1.0 / 0.9375, 0.25 / 0.9375, 0.25 / 0.9375, 1.0 / 0.9375),
        _ => return,
    };
    let shape = spec.l.shape().to_vec();
    for f in 0..shape[0] {
        for t in 0..shape[1] {
            for ri in 0..2 {
                let l = spec.l[[f, t, ri]];
                let r = spec.r[[f, t, ri]];
                // NOTE component order: mid_side is (L+R)/2, L−R — a0*l + a1*r / b0*l + b1*r
                spec.l[[f, t, ri]] = a0 * l + a1 * r;
                spec.r[[f, t, ri]] = b0 * l + b1 * r;
            }
        }
    }
}

/// Inverse channel transforms applied to each band's iSTFT output (f64 waves).
/// v5.0 global (reverse / mid_side / mid_side_b2) and v5.1 per-band variants —
/// constants copied verbatim (they are exact algebraic inverses).
pub fn inverse_channel_transform(
    params: &VrParams,
    cc: Option<&str>,
    wl: Vec<f64>,
    wr: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    if params.is_v51 {
        match cc {
            Some("mid_side_c") => {
                let l: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| a / 1.0625 - b / 4.25).collect();
                let r: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| b / 1.0625 + a / 4.25).collect();
                (l, r)
            }
            Some("mid_side") => {
                let l: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| a + b / 2.0).collect();
                let r: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| a - b / 2.0).collect();
                (l, r)
            }
            Some("stereo_n") => {
                let l: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| a - b * 0.25).collect();
                let r: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| b - a * 0.25).collect();
                (l, r)
            }
            _ => (wl, wr),
        }
    } else if params.reverse {
        (wl.into_iter().rev().collect(), wr.into_iter().rev().collect())
    } else if params.mid_side {
        let l: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| a + b / 2.0).collect();
        let r: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| a - b / 2.0).collect();
        (l, r)
    } else if params.mid_side_b2 {
        let l: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| b / 1.25 + 0.4 * a).collect();
        let r: Vec<f64> = wl.iter().zip(&wr).map(|(a, b)| a / 1.25 - 0.4 * b).collect();
        (l, r)
    } else {
        (wl, wr)
    }
}

// ─── Spectral filters ────────────────────────────────────────────

/// v5.0 `fft_lp_filter`: stepped ramp 1→0 over [start, stop), rows ≥ stop zeroed.
pub fn fft_lp_filter(spec: &mut VrSpec, start: i64, stop: i64) {
    let bins = spec.l.shape()[0] as i64;
    let frames = spec.l.shape()[1];
    let mut g = 1.0f64;
    for b in start.max(0)..stop.min(bins) {
        g -= 1.0 / (stop - start) as f64;
        scale_row(spec, b as usize, frames, g);
    }
    for b in stop.max(0)..bins {
        scale_row(spec, b as usize, frames, 0.0);
    }
}

/// v5.0 `fft_hp_filter`: called with (hpf_start, hpf_stop - 1); stepped ramp 1→0
/// walking DOWN from start to stop+1, rows [0, stop] zeroed.
pub fn fft_hp_filter(spec: &mut VrSpec, start: i64, stop: i64) {
    let bins = spec.l.shape()[0] as i64;
    let frames = spec.l.shape()[1];
    let mut g = 1.0f64;
    let mut b = start;
    while b > stop {
        g -= 1.0 / (start - stop) as f64;
        if b >= 0 && b < bins {
            scale_row(spec, b as usize, frames, g);
        }
        b -= 1;
    }
    let upper = (stop + 1).clamp(0, bins);
    for bb in 0..upper {
        scale_row(spec, bb as usize, frames, 0.0);
    }
}

/// v5.1 `get_lp_filter_mask` per-row gain: ones(start-1) ++ linspace(1,0,stop-start+1)
/// ++ zeros(n_bins-stop).
pub fn lp_mask_gain(n_bins: usize, start: i64, stop: i64, row: usize) -> f64 {
    let row = row as i64;
    if row < start - 1 {
        1.0
    } else if row <= stop - 1 {
        let n = stop - start + 1; // linspace point count
        let i = row - (start - 1);
        if n <= 1 { 1.0 } else { 1.0 - i as f64 / (n - 1) as f64 }
    } else if (row as usize) < n_bins {
        0.0
    } else {
        0.0
    }
}

/// v5.1 `get_hp_filter_mask` per-row gain: zeros(stop+1) ++ linspace(0,1,1+start-stop)
/// ++ ones(n_bins-start-2). Callers pass stop = hpf_stop - 1 (matching the reference).
pub fn hp_mask_gain(_n_bins: usize, start: i64, stop: i64, row: usize) -> f64 {
    let row = row as i64;
    if row <= stop {
        0.0
    } else if row <= start + 1 {
        let n = 1 + start - stop; // linspace point count
        let i = row - (stop + 1);
        if n <= 1 { 1.0 } else { i as f64 / (n - 1) as f64 }
    } else {
        1.0
    }
}

pub fn apply_lp_mask(spec: &mut VrSpec, start: i64, stop: i64) {
    let bins = spec.l.shape()[0];
    let frames = spec.l.shape()[1];
    for row in 0..bins {
        let g = lp_mask_gain(bins, start, stop, row);
        if g != 1.0 {
            scale_row(spec, row, frames, g);
        }
    }
}

pub fn apply_hp_mask(spec: &mut VrSpec, start: i64, stop: i64) {
    let bins = spec.l.shape()[0];
    let frames = spec.l.shape()[1];
    for row in 0..bins {
        let g = hp_mask_gain(bins, start, stop, row);
        if g != 1.0 {
            scale_row(spec, row, frames, g);
        }
    }
}

fn scale_row(spec: &mut VrSpec, row: usize, frames: usize, g: f64) {
    for t in 0..frames {
        for ri in 0..2 {
            spec.l[[row, t, ri]] = (spec.l[[row, t, ri]] as f64 * g) as f32;
            spec.r[[row, t, ri]] = (spec.r[[row, t, ri]] as f64 * g) as f32;
        }
    }
}

// ─── Analysis: multiband cascade + combine ───────────────────────

/// Full VR analysis: resample cascade (top band = input, each lower band resampled
/// from the one above with `polyphase`) → per-band STFT (+ channel transforms) →
/// combine_spectrograms (crop/stack rows + pre-filter). Input = 44100 stereo
/// (`params.bands.last().sr` must equal the audio's rate — the caller guarantees it).
/// Returns the combined [bins+1, frames, 2]×2 spectrum.
pub fn vr_analyze(params: &VrParams, left: &[f32], right: &[f32]) -> VrSpec {
    let bands_n = params.bands.len();
    // Band waves, index 0 = band 1 (lowest). Built top→down like the reference.
    let mut waves: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; bands_n];
    let mut specs: Vec<Option<VrSpec>> = (0..bands_n).map(|_| None).collect();

    for d in (1..=bands_n).rev() {
        let bp = &params.bands[d - 1];
        let (wl, wr) = if d == bands_n {
            (left.to_vec(), right.to_vec())
        } else {
            let upper = &params.bands[d]; // band d+1 (1-indexed)
            let (ul, ur) = waves[d].as_ref().expect("upper band wave built first");
            let up = bp.sr as usize;
            let down = upper.sr as usize;
            (resample_poly_f32(ul, up, down), resample_poly_f32(ur, up, down))
        };

        // wave_to_spectrogram: v5.0 applies the GLOBAL wave transform per band;
        // v5.1 applies per-band spectrum transforms after the STFT.
        let spec = if !params.is_v51 {
            let (tl, tr) = v4_wave_transform(params, &wl, &wr);
            VrSpec {
                l: librosa_stft(&tl, bp.n_fft, bp.hl),
                r: librosa_stft(&tr, bp.n_fft, bp.hl),
            }
        } else {
            let mut s = VrSpec {
                l: librosa_stft(&wl, bp.n_fft, bp.hl),
                r: librosa_stft(&wr, bp.n_fft, bp.hl),
            };
            v51_convert_channels(&mut s, bp.convert_channels.as_deref());
            s
        };
        waves[d - 1] = Some((wl, wr));
        specs[d - 1] = Some(spec);
    }
    drop(waves);

    // combine_spectrograms
    let l_frames = specs
        .iter()
        .map(|s| s.as_ref().unwrap().l.shape()[1])
        .min()
        .unwrap_or(0);
    let mut combined = VrSpec {
        l: Array3::<f32>::zeros((params.bins + 1, l_frames, 2)),
        r: Array3::<f32>::zeros((params.bins + 1, l_frames, 2)),
    };
    let mut offset = 0usize;
    for d in 1..=bands_n {
        let bp = &params.bands[d - 1];
        let s = specs[d - 1].as_ref().unwrap();
        let h = bp.crop_stop - bp.crop_start;
        for i in 0..h {
            for t in 0..l_frames {
                for ri in 0..2 {
                    combined.l[[offset + i, t, ri]] = s.l[[bp.crop_start + i, t, ri]];
                    combined.r[[offset + i, t, ri]] = s.r[[bp.crop_start + i, t, ri]];
                }
            }
        }
        offset += h;
    }
    debug_assert!(offset <= params.bins, "Too much bins");

    // pre-filter (lowpass at the very top of the combined spectrum)
    if params.pre_filter_start > 0 {
        if params.is_v51 {
            apply_lp_mask(&mut combined, params.pre_filter_start, params.pre_filter_stop);
        } else if bands_n == 1 {
            fft_lp_filter(&mut combined, params.pre_filter_start, params.pre_filter_stop);
        } else {
            let mut gp = 1.0f64;
            for b in (params.pre_filter_start + 1)..params.pre_filter_stop {
                let g = 10.0f64.powf(-(b - params.pre_filter_start) as f64 * (3.5 - gp) / 20.0);
                gp = g;
                if b >= 0 && (b as usize) < params.bins + 1 {
                    scale_row(&mut combined, b as usize, l_frames, g);
                }
            }
        }
    }
    combined
}

/// |spec| and angle(spec) as [2, bins, frames] f32 (channel-major like the
/// reference's (2, bins+1, T) numpy arrays).
pub fn vr_mag_phase(spec: &VrSpec) -> (Array3<f32>, Array3<f32>) {
    let bins = spec.l.shape()[0];
    let frames = spec.l.shape()[1];
    let mut mag = Array3::<f32>::zeros((2, bins, frames));
    let mut phase = Array3::<f32>::zeros((2, bins, frames));
    for (ch, s) in [&spec.l, &spec.r].into_iter().enumerate() {
        for f in 0..bins {
            for t in 0..frames {
                let re = s[[f, t, 0]];
                let im = s[[f, t, 1]];
                mag[[ch, f, t]] = re.hypot(im);
                phase[[ch, f, t]] = im.atan2(re);
            }
        }
    }
    (mag, phase)
}

/// spec_utils.make_padding → (pad_left, pad_right, roi_size).
pub fn make_padding(width: usize, cropsize: usize, offset: usize) -> (usize, usize, usize) {
    let left = offset;
    let mut roi_size = cropsize.saturating_sub(offset * 2);
    if roi_size == 0 {
        roi_size = cropsize;
    }
    let right = roi_size - (width % roi_size) + left;
    (left, right, roi_size)
}

/// Zero-pad the mag time axis by (pad_l, pad_r) and divide by the GLOBAL max
/// (the reference normalizes the padded array in place; zero-max guarded).
pub fn pad_and_normalize_mag(mag: &Array3<f32>, pad_l: usize, pad_r: usize) -> Array3<f32> {
    let bins = mag.shape()[1];
    let frames = mag.shape()[2];
    let mut out = Array3::<f32>::zeros((2, bins, pad_l + frames + pad_r));
    let mut maxv = 0.0f32;
    for ch in 0..2 {
        for f in 0..bins {
            for t in 0..frames {
                let v = mag[[ch, f, t]];
                out[[ch, f, pad_l + t]] = v;
                if v > maxv {
                    maxv = v;
                }
            }
        }
    }
    if maxv > 0.0 {
        out.mapv_inplace(|v| v / maxv);
    }
    out
}

/// Copy one model window [2, bins, window_size] starting at time `start` from the
/// padded mag into `dst` (a flat [2*bins*window_size] slice, ONNX row-major layout).
pub fn copy_mag_window(padded: &Array3<f32>, start: usize, window_size: usize, dst: &mut [f32]) {
    let bins = padded.shape()[1];
    let total_t = padded.shape()[2];
    debug_assert_eq!(dst.len(), 2 * bins * window_size);
    for ch in 0..2 {
        for f in 0..bins {
            let base = (ch * bins + f) * window_size;
            for i in 0..window_size {
                let t = start + i;
                dst[base + i] = if t < total_t { padded[[ch, f, t]] } else { 0.0 };
            }
        }
    }
}

/// spec_utils.adjust_aggr — in-place mask exponentiation. `value` is the UI
/// aggression / 100 (default 0.05). No aggr_correction (no shipped JSON sets one).
pub fn adjust_aggr(mask: &mut Array3<f32>, is_non_accom_stem: bool, value: f64, split_bin: usize) {
    let mut aggr = value * 2.0;
    if aggr == 0.0 {
        return;
    }
    if is_non_accom_stem {
        aggr = 1.0 - aggr;
    }
    let bins = mask.shape()[1];
    let frames = mask.shape()[2];
    let low_exp = (1.0 + aggr / 3.0) as f32;
    let high_exp = (1.0 + aggr) as f32;
    for ch in 0..2 {
        for f in 0..bins {
            let e = if f < split_bin { low_exp } else { high_exp };
            for t in 0..frames {
                let v = mask[[ch, f, t]];
                mask[[ch, f, t]] = v.powf(e);
            }
        }
    }
}

/// spec_utils.merge_artifacts (post-process): pushes the mask toward 1 across
/// sustained high-confidence frame runs. Exact port including the run-merge
/// (`s = old_e - fade*2`) and edge (`s=0`/`e=T`) quirks; an empty index set is
/// the reference's caught IndexError → mask unchanged.
pub fn merge_artifacts(mask: &mut Array3<f32>, thres: f32, min_range: usize, fade_size: usize) {
    assert!(min_range >= fade_size * 2, "min_range must be >= fade_size * 2");
    let bins = mask.shape()[1];
    let frames = mask.shape()[2];

    // idx = frames whose min over (ch, bins) > thres
    let mut idx: Vec<usize> = Vec::new();
    for t in 0..frames {
        let mut mn = f32::MAX;
        for ch in 0..2 {
            for f in 0..bins {
                let v = mask[[ch, f, t]];
                if v < mn {
                    mn = v;
                }
            }
        }
        if mn > thres {
            idx.push(t);
        }
    }
    if idx.is_empty() {
        return; // reference: IndexError on idx[0], caught → unchanged
    }

    // contiguous runs [start, end] (end inclusive, matching the numpy construction)
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut run_start = idx[0];
    let mut prev = idx[0];
    for &i in idx.iter().skip(1) {
        if i != prev + 1 {
            runs.push((run_start, prev));
            run_start = i;
        }
        prev = i;
    }
    runs.push((run_start, prev));

    // weight ramp over runs longer than min_range (strict >)
    let mut weight = vec![0.0f32; frames]; // per-frame (broadcast over ch/bins)
    let fade = fade_size as i64;
    let frames_i = frames as i64;
    let mut old_e: Option<i64> = None;
    for &(s0, e0) in runs.iter().filter(|&&(s, e)| e - s > min_range) {
        let mut s = s0 as i64;
        let e_orig = e0 as i64;
        let mut e = e_orig;
        if let Some(oe) = old_e {
            if s - oe < fade {
                s = oe - fade * 2;
            }
        }
        if s != 0 {
            // weight[s : s+fade] = linspace(0, 1, fade)
            for i in 0..fade {
                let pos = s + i;
                if pos >= 0 && pos < frames_i {
                    weight[pos as usize] = i as f32 / (fade - 1) as f32;
                }
            }
        } else {
            s -= fade; // python: s becomes -fade so s+fade == 0 below
        }
        if e != frames_i {
            // weight[e-fade : e] = linspace(1, 0, fade)
            for i in 0..fade {
                let pos = e - fade + i;
                if pos >= 0 && pos < frames_i {
                    weight[pos as usize] = 1.0 - i as f32 / (fade - 1) as f32;
                }
            }
        } else {
            e += fade;
        }
        // weight[s+fade : e-fade] = 1
        let lo = (s + fade).max(0);
        let hi = (e - fade).min(frames_i);
        for pos in lo..hi {
            weight[pos as usize] = 1.0;
        }
        old_e = Some(e);
    }

    // y_mask += weight * (1 - y_mask)
    for t in 0..frames {
        let w = weight[t];
        if w == 0.0 {
            continue;
        }
        for ch in 0..2 {
            for f in 0..bins {
                let v = mask[[ch, f, t]];
                mask[[ch, f, t]] = v + w * (1.0 - v);
            }
        }
    }
}

/// y = mask · mag · e^{iφ}, v = (1−mask) · mag · e^{iφ} (+ nan_to_num like the
/// audio-separator reference).
pub fn vr_apply_mask(
    mask: &Array3<f32>,
    mag: &Array3<f32>,
    phase: &Array3<f32>,
) -> (VrSpec, VrSpec) {
    let bins = mag.shape()[1];
    let frames = mag.shape()[2];
    let mut y = VrSpec {
        l: Array3::<f32>::zeros((bins, frames, 2)),
        r: Array3::<f32>::zeros((bins, frames, 2)),
    };
    let mut v = VrSpec {
        l: Array3::<f32>::zeros((bins, frames, 2)),
        r: Array3::<f32>::zeros((bins, frames, 2)),
    };
    let clean = |x: f32| if x.is_finite() { x } else { 0.0 };
    for ch in 0..2 {
        for f in 0..bins {
            for t in 0..frames {
                let m = mask[[ch, f, t]];
                let a = mag[[ch, f, t]];
                let p = phase[[ch, f, t]];
                let (sin, cos) = p.sin_cos();
                let (yre, yim) = (m * a * cos, m * a * sin);
                let (vre, vim) = ((1.0 - m) * a * cos, (1.0 - m) * a * sin);
                let (ty, tv) = if ch == 0 { (&mut y.l, &mut v.l) } else { (&mut y.r, &mut v.r) };
                ty[[f, t, 0]] = clean(yre);
                ty[[f, t, 1]] = clean(yim);
                tv[[f, t, 0]] = clean(vre);
                tv[[f, t, 1]] = clean(vim);
            }
        }
    }
    (y, v)
}

// ─── Synthesis: cmb_spectrogram_to_wave ──────────────────────────

/// Reconstruct a stereo wave from a combined masked spectrum. Per band: scatter the
/// band's rows into a zeroed n_fft/2+1 scratch spectrum, apply hp/lp edge filters
/// (v5.0 stepped loops vs v5.1 linspace masks), iSTFT (f64), inverse channel
/// transform, then cascade lowest→highest with polyphase upsampling between band
/// rates. Output f32 at the model rate (44100 for all shipped configs).
pub fn vr_synthesize(params: &VrParams, spec_m: &VrSpec) -> (Vec<f32>, Vec<f32>) {
    let bands_n = params.bands.len();
    let frames = spec_m.l.shape()[1];
    let mut offset = 0usize;
    let mut wave: Option<(Vec<f64>, Vec<f64>)> = None;

    for d in 1..=bands_n {
        let bp = &params.bands[d - 1];
        let band_bins = bp.n_fft / 2 + 1;
        let mut spec_s = VrSpec {
            l: Array3::<f32>::zeros((band_bins, frames, 2)),
            r: Array3::<f32>::zeros((band_bins, frames, 2)),
        };
        let h = bp.crop_stop - bp.crop_start;
        for i in 0..h {
            for t in 0..frames {
                for ri in 0..2 {
                    spec_s.l[[bp.crop_start + i, t, ri]] = spec_m.l[[offset + i, t, ri]];
                    spec_s.r[[bp.crop_start + i, t, ri]] = spec_m.r[[offset + i, t, ri]];
                }
            }
        }
        offset += h;

        let is_top = d == bands_n;
        let is_bottom = d == 1;
        if is_top {
            // top band: hpf only, and only when hpf_start > 0 (1band configs carry -1)
            let hs = bp.hpf_start.unwrap_or(-1);
            if hs > 0 {
                let he = bp.hpf_stop.expect("top band hpf_stop") - 1;
                if params.is_v51 {
                    apply_hp_mask(&mut spec_s, hs, he);
                } else {
                    fft_hp_filter(&mut spec_s, hs, he);
                }
            }
        } else if is_bottom {
            let ls = bp.lpf_start.expect("bottom band lpf_start");
            let le = bp.lpf_stop.expect("bottom band lpf_stop");
            if params.is_v51 {
                apply_lp_mask(&mut spec_s, ls, le);
            } else {
                fft_lp_filter(&mut spec_s, ls, le);
            }
        } else {
            let hs = bp.hpf_start.expect("mid band hpf_start");
            let he = bp.hpf_stop.expect("mid band hpf_stop") - 1;
            let ls = bp.lpf_start.expect("mid band lpf_start");
            let le = bp.lpf_stop.expect("mid band lpf_stop");
            if params.is_v51 {
                apply_hp_mask(&mut spec_s, hs, he);
                apply_lp_mask(&mut spec_s, ls, le);
            } else {
                fft_hp_filter(&mut spec_s, hs, he);
                fft_lp_filter(&mut spec_s, ls, le);
            }
        }

        // spectrogram_to_wave: per-channel iSTFT + inverse channel transform
        let wl = librosa_istft_f64(&spec_s.l, bp.n_fft, bp.hl);
        let wr = librosa_istft_f64(&spec_s.r, bp.n_fft, bp.hl);
        let (wl, wr) =
            inverse_channel_transform(params, bp.convert_channels.as_deref(), wl, wr);

        if bands_n == 1 {
            wave = Some((wl, wr));
        } else if is_bottom {
            let next_sr = params.bands[d].sr as usize; // band 2
            let cur_sr = bp.sr as usize;
            wave = Some((
                resample_poly(&wl, next_sr, cur_sr),
                resample_poly(&wr, next_sr, cur_sr),
            ));
        } else if is_top {
            let (al, ar) = wave.take().expect("lower bands accumulated");
            let sl: Vec<f64> = al.iter().zip(&wl).map(|(a, b)| a + b).collect();
            let sr_ = ar.iter().zip(&wr).map(|(a, b)| a + b).collect();
            wave = Some((sl, sr_));
        } else {
            let (al, ar) = wave.take().expect("lower bands accumulated");
            let sl: Vec<f64> = al.iter().zip(&wl).map(|(a, b)| a + b).collect();
            let sr_: Vec<f64> = ar.iter().zip(&wr).map(|(a, b)| a + b).collect();
            let next_sr = params.bands[d].sr as usize; // band d+1
            let cur_sr = bp.sr as usize;
            wave = Some((
                resample_poly(&sl, next_sr, cur_sr),
                resample_poly(&sr_, next_sr, cur_sr),
            ));
        }
    }

    let (l, r) = wave.unwrap_or((vec![], vec![]));
    (
        l.into_iter().map(|v| v as f32).collect(),
        r.into_iter().map(|v| v as f32).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic multi-tone used by the scipy/librosa reference tests below
    /// (values generated by scratchpad gen_dsp_refs.py in the converter venv:
    /// scipy 1.x resample_poly window=('kaiser',5.0), librosa 0.11).
    fn ref_signal_f64(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                (0.05 * t).sin() + 0.5 * (0.31 * t + 1.0).sin() + 0.25 * (1.7 * t + 2.0).sin()
            })
            .collect()
    }

    #[test]
    fn resample_matches_scipy() {
        let x = ref_signal_f64(2000);
        // (up, down, len, head[..6], sum, absmax, mid)
        let cases: [(usize, usize, usize, [f64; 6], f64, f64, f64); 4] = [
            (1, 3, 667,
             [3.377645752735e-01, 6.373432659381e-01, 4.241435090137e-01,
              1.389369884235e-01, 6.019971296939e-02, 3.883540311397e-01],
             1.245407346676e0, 1.499201722909e0, -1.494211587550e-01),
            (1, 2, 1000,
             [3.579530609052e-01, 5.987286949140e-01, 5.964922354695e-01,
              4.314637335020e-01, 2.210320450246e-01, 7.867717850980e-02],
             1.934276646207e0, 1.516549471927e0, -2.742683891146e-01),
            (2, 1, 4000,
             [6.483952010268e-01, 6.304424754408e-01, 4.008199150959e-01,
              2.837011364591e-01, 4.062472787877e-01, 6.325784352969e-01],
             6.873292601680e0, 1.729229659156e0, -4.229229452022e-01),
            (3, 1, 6000,
             [6.484526858522e-01, 6.739930711075e-01, 5.613871558053e-01,
              4.008554506193e-01, 2.954929226093e-01, 3.011595289280e-01],
             1.014196822364e1, 1.729382967609e0, -4.229604403157e-01),
        ];
        for (up, down, len, head, sum, absmax, mid) in cases {
            let y = resample_poly(&x, up, down);
            assert_eq!(y.len(), len, "len up={up} down={down}");
            for (i, &h) in head.iter().enumerate() {
                assert!((y[i] - h).abs() < 1e-10, "up={up} down={down} y[{i}]={} vs {h}", y[i]);
            }
            let s: f64 = y.iter().sum();
            let m = y.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            assert!((s - sum).abs() < 1e-8, "sum up={up} down={down}: {s} vs {sum}");
            assert!((m - absmax).abs() < 1e-10, "absmax up={up} down={down}: {m} vs {absmax}");
            assert!((y[len / 2] - mid).abs() < 1e-10, "mid up={up} down={down}");
        }
    }

    #[test]
    fn librosa_stft_matches_reference() {
        let x: Vec<f32> = ref_signal_f64(2000).into_iter().map(|v| v as f32).collect();
        let spec = librosa_stft(&x, 512, 128);
        assert_eq!(spec.shape(), &[257, 16, 2]);
        let cases: [(usize, usize, f32, f32); 4] = [
            (0, 0, 2.240422249e1, 0.0),
            (10, 5, 1.932489499e-2, -4.141386971e-2),
            (100, 7, 1.579767122e-4, -4.930498471e-5),
            (256, 10, -9.970989595e-6, 0.0),
        ];
        for (f, t, re, im) in cases {
            let dr = (spec[[f, t, 0]] - re).abs();
            let di = (spec[[f, t, 1]] - im).abs();
            assert!(dr < 2e-4 && di < 2e-4,
                "S[{f},{t}] = {}+{}j vs {re}+{im}j", spec[[f, t, 0]], spec[[f, t, 1]]);
        }
        let mut abs_sum = 0.0f64;
        for f in 0..257 {
            for t in 0..16 {
                abs_sum += (spec[[f, t, 0]] as f64).hypot(spec[[f, t, 1]] as f64);
            }
        }
        assert!((abs_sum - 8.046323242e3).abs() / 8.046323242e3 < 1e-6,
            "abs sum {abs_sum}");

        // istft reference (librosa dtype=f64): len 1920, sum, mid value
        let y = librosa_istft_f64(&spec, 512, 128);
        assert_eq!(y.len(), 1920);
        let s: f64 = y.iter().sum();
        assert!((s - 2.328012398903e1).abs() < 1e-4, "istft sum {s}");
        assert!((y[960] - -7.515019863648e-1).abs() < 1e-6, "istft mid {}", y[960]);
    }

    #[test]
    fn resample_identity() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y = resample_poly(&x, 3, 3);
        assert_eq!(x, y);
    }

    #[test]
    fn resample_lengths() {
        // scipy: n_out = ceil(n_in * up / down)
        let x = vec![0.0f64; 1000];
        assert_eq!(resample_poly(&x, 1, 3).len(), 334);
        assert_eq!(resample_poly(&x, 1, 2).len(), 500);
        assert_eq!(resample_poly(&x, 2, 1).len(), 2000);
        assert_eq!(resample_poly(&x, 3, 1).len(), 3000);
    }

    #[test]
    fn resample_dc_preserved() {
        // A constant signal must stay ~constant through the kaiser filter (DC gain 1).
        let x = vec![1.0f64; 2000];
        let y = resample_poly(&x, 1, 2);
        let mid = &y[100..y.len() - 100];
        for &v in mid {
            assert!((v - 1.0).abs() < 1e-9, "DC drift: {v}");
        }
    }

    #[test]
    fn librosa_stft_frame_count_and_roundtrip() {
        let n_fft = 512;
        let hop = 128;
        let n = 5000;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let spec = librosa_stft(&signal, n_fft, hop);
        assert_eq!(spec.shape()[0], n_fft / 2 + 1);
        assert_eq!(spec.shape()[1], 1 + n / hop); // librosa center frame count

        let rec = librosa_istft_f64(&spec, n_fft, hop);
        assert_eq!(rec.len(), hop * (n / hop)); // hop*(frames-1)
        // interior samples must reconstruct (librosa roundtrip is exact where NOLA holds)
        for i in n_fft..rec.len().saturating_sub(n_fft) {
            assert!(
                (rec[i] - signal[i] as f64).abs() < 1e-5,
                "roundtrip err at {i}: {} vs {}", rec[i], signal[i]
            );
        }
    }

    #[test]
    fn make_padding_matches_reference() {
        // v4: window 512, offset 128 → roi 256
        let (l, r, roi) = make_padding(22050, 512, 128);
        assert_eq!(roi, 256);
        assert_eq!(l, 128);
        assert_eq!(r, 256 - (22050 % 256) + 128);
        // v5: offset 64 → roi 384
        let (_, _, roi5) = make_padding(1000, 512, 64);
        assert_eq!(roi5, 384);
    }

    #[test]
    fn merge_artifacts_empty_noop() {
        let mut mask = Array3::<f32>::zeros((2, 4, 100));
        let before = mask.clone();
        merge_artifacts(&mut mask, 0.2, 64, 32);
        assert_eq!(mask, before);
    }

    #[test]
    fn merge_artifacts_long_run() {
        // 300 frames all above threshold → single run (0, 299). Python semantics:
        // the run end is an INCLUSIVE index (max T-1), so `e != T` is always true
        // and a fade-OUT is always written over [e-fade, e); s==0 hits the
        // `s -= fade` branch so the interior plateau starts at frame 0.
        let fade = 32usize;
        let mut mask = Array3::<f32>::from_elem((2, 4, 300), 0.5f32);
        merge_artifacts(&mut mask, 0.2, 64, fade);
        let e = 299usize;
        for t in 0..(e - fade) {
            let v = mask[[0, 0, t]];
            assert!((v - 1.0).abs() < 1e-6, "plateau frame {t}: {v}");
        }
        for i in 0..fade {
            let w = 1.0 - i as f32 / (fade - 1) as f32; // linspace(1,0,fade)
            let expect = 0.5 + 0.5 * w;
            let v = mask[[0, 0, e - fade + i]];
            assert!((v - expect).abs() < 1e-6, "fade frame {}: {v} vs {expect}", e - fade + i);
        }
        // the inclusive end frame itself is never written by any slice → unchanged
        assert!((mask[[0, 0, e]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn aggr_default_exponents() {
        // default aggression 5 → value 0.05 → aggr 0.1 (accom primary):
        // below split: m^(1.0333...), above: m^1.1
        let mut mask = Array3::<f32>::from_elem((2, 10, 4), 0.5f32);
        adjust_aggr(&mut mask, false, 0.05, 5);
        let below = 0.5f32.powf(1.0 + 0.1f32 / 3.0);
        let above = 0.5f32.powf(1.1);
        assert!((mask[[0, 0, 0]] - below).abs() < 1e-6);
        assert!((mask[[0, 9, 0]] - above).abs() < 1e-6);
    }
}
