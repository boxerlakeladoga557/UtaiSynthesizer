// ② Vocal pitch — evalF0Cents, THE single f0 evaluator (S50 Phase-5 foundation, §10, Option A).
// One source of truth for the sounding pitch line, feeding: the editor OVERLAY draw + the light PREVIEW
// oscillator (→ centsToHz) + (Phase 6) the Rust render array. Option A = TS computes this array and Rust
// consumes it (+transpose → Hz → resample), so "what you see == hear == render" holds by construction —
// no two-port drift. Returns WRITTEN-pitch cents (transpose is applied ONLY in the Rust render, §9.3).
//
// ⚠ FOUNDATION SCOPE (S50): the layer STRUCTURE + the certain layers (base / pitchPoints within a note /
// pitchDev) are solid; the vibrato constants + the CROSS-note-boundary pre-onset slide (a pitchPoint x<0
// rendering into the previous note's tail) are distilled from OpenUTAU semantics but NOT verified against
// OpenUTAU source — VERIFY + thread neighbor context before freezing the composed golden-vector gate
// (§10.2). See lib/interpolateShape.ts for the easings.
import type { Note, PitchCurve } from "../types/project";
import { evalPitchPoints } from "./interpolateShape";
import { ticksToMs } from "./audio/laneOps";

export interface F0EvalOpts {
  /** Tempo (BPM) — vibrato period is in ms, so tick↔ms needs it. */
  tempo: number;
}

/** Linear-interpolate a PitchCurve (parallel strictly-increasing xs / ys) at `x`; hold flat outside; 0 if empty. */
function evalCurveAt(c: PitchCurve | undefined, x: number): number {
  if (!c || c.xs.length === 0) return 0;
  const { xs, ys } = c;
  const n = xs.length;
  if (x <= xs[0]!) return ys[0]!;
  if (x >= xs[n - 1]!) return ys[n - 1]!;
  let i = 0;
  while (i < n - 1 && xs[i + 1]! <= x) i++;
  const span = xs[i + 1]! - xs[i]!;
  const t = span > 0 ? (x - xs[i]!) / span : 0;
  return ys[i]! + (ys[i + 1]! - ys[i]!) * t;
}

/** Index of the note covering segment-relative `relTick` (half-open [tick, tick+duration)); -1 = rest.
 *  Notes are tick-sorted (normalizeNotesArray) → a linear scan suits editor note counts (binary search is
 *  a trivial later optimization). Returns the LAST match so an abutting note's onset wins over the prior end. */
function findNoteAt(notes: readonly Note[], relTick: number): number {
  let found = -1;
  for (let i = 0; i < notes.length; i++) {
    const n = notes[i]!;
    if (relTick >= n.tick && relTick < n.tick + n.duration) found = i;
    else if (n.tick > relTick) break; // sorted → no later note can cover an earlier tick
  }
  return found;
}

/** ④ Vibrato tail LFO (OpenUTAU UVibrato semantics; ⚠ verify vs OpenUTAU before freezing golden vectors).
 *  Tail-anchored to the last `length` fraction of the note: depth cents, period ms, in/out linear fade
 *  fractions of the span, shift phase (cycles), drift cents linear across the span. */
function evalVibrato(note: Note, noteRel: number, tempo: number): number {
  const v = note.vibrato!;
  const dur = note.duration;
  if (v.length <= 0 || dur <= 0 || v.period <= 0) return 0;
  const p = noteRel / dur; // note-normalized position [0,1]
  const nStart = 1 - v.length; // vibrato span starts here
  if (p < nStart) return 0; // before the span → no vibrato
  const inSpan = (p - nStart) / v.length; // [0,1] within the span
  const elapsedMs = ticksToMs((p - nStart) * dur, tempo);
  const base = v.depth * Math.sin(2 * Math.PI * (elapsedMs / v.period + v.shift));
  let env = 1;
  if (v.in > 0 && inSpan < v.in) env = inSpan / v.in; // linear fade-in
  if (v.out > 0 && inSpan > 1 - v.out) env = Math.min(env, (1 - inSpan) / v.out); // linear fade-out
  return base * env + v.drift * inSpan; // drift = slow linear pitch drift across the span
}

/**
 * ★ evalF0Cents at one segment-relative tick → { WRITTEN-pitch cents, voiced }. Layered (§3.2):
 *   ① base = note.pitch*100 + detune   ② pitchPoints transition   ③ pitchDev additive   ④ vibrato.
 * A rest (no note) → voiced:false (the overlay breaks the line; the Rust render's uv=(f0<30) mirrors it).
 */
export function evalF0CentsAt(
  notes: readonly Note[],
  pitchDev: PitchCurve | undefined,
  relTick: number,
  opts: F0EvalOpts,
): { cents: number; voiced: boolean } {
  const dev = evalCurveAt(pitchDev, relTick); // ③ (segment-relative; applies over rests too, but they're unvoiced)
  const idx = findNoteAt(notes, relTick);
  if (idx < 0) return { cents: dev, voiced: false };
  const note = notes[idx]!;
  const noteRel = relTick - note.tick;
  let cents = note.pitch * 100 + (note.detune ?? 0); // ①
  if (note.pitchPoints && note.pitchPoints.length > 0) cents += evalPitchPoints(note.pitchPoints, noteRel); // ②
  cents += dev;
  if (note.vibrato) cents += evalVibrato(note, noteRel, opts.tempo); // ④
  return { cents, voiced: true };
}

export interface F0Frames {
  cents: Float32Array;
  voiced: Uint8Array;
}

/**
 * Batch: sample evalF0Cents at `frameCount` frames, `ticksPerFrame` apart from `frameStartTick` (segment-
 * relative). The canonical per-frame f0 array — the overlay samples it to draw the line, and (Option A)
 * Phase 6 hands it to the Rust render (which adds transpose, converts →Hz with the voiced mask, resamples).
 */
export function evalF0CentsFrames(
  notes: readonly Note[],
  pitchDev: PitchCurve | undefined,
  frame: { frameStartTick: number; ticksPerFrame: number; frameCount: number },
  opts: F0EvalOpts,
): F0Frames {
  const cents = new Float32Array(frame.frameCount);
  const voiced = new Uint8Array(frame.frameCount);
  for (let f = 0; f < frame.frameCount; f++) {
    const r = evalF0CentsAt(notes, pitchDev, frame.frameStartTick + f * frame.ticksPerFrame, opts);
    cents[f] = r.cents;
    voiced[f] = r.voiced ? 1 : 0;
  }
  return { cents, voiced };
}
