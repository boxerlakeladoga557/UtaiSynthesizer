// ② Vocal pitch — OpenUTAU MusicMath.InterpolateShape port (S50 Phase-5 foundation, §10.2 step 2).
// PURE easings + a pitch-point polyline evaluator. These 4 shapes are standard easings (not OpenUTAU-
// specific), so they are certain; the polyline convention (the LATER point owns its incoming segment,
// flat hold outside the point range) matches OpenUTAU. Used by lib/f0eval.ts (evalF0Cents) — the single
// f0 source that feeds overlay + preview + (Phase 6) Rust render.
import type { PitchPoint } from "../types/project";

export type PitchShape = PitchPoint["shape"]; // "linear" | "sineIn" | "sineOut" | "sineInOut"

/**
 * Easing f(t) ∈ [0,1] for t ∈ [0,1] — the shape of the segment leading UP TO a pitch point:
 *   linear     f=t
 *   sineIn     f=1−cos(t·π/2)     (slow start / ease-in)
 *   sineOut    f=sin(t·π/2)       (slow end / ease-out)
 *   sineInOut  f=(1−cos(t·π))/2   (S-curve; OpenUTAU default)
 */
export function interpShape(t: number, shape: PitchShape): number {
  const x = t <= 0 ? 0 : t >= 1 ? 1 : t;
  switch (shape) {
    case "sineIn": return 1 - Math.cos((x * Math.PI) / 2);
    case "sineOut": return Math.sin((x * Math.PI) / 2);
    case "sineInOut": return (1 - Math.cos(x * Math.PI)) / 2;
    case "linear":
    default: return x;
  }
}

/**
 * Evaluate a pitch-point polyline at `x` (same X unit as the points — note-relative ticks for
 * Note.pitchPoints). Each segment interpolates with the LATER point's `shape`; before the first point /
 * after the last point holds that endpoint's `y` (flat). Points MUST be sorted by x ascending
 * (`normalizeNote` guarantees this). Empty → 0. Returns the cents OFFSET the polyline contributes.
 *
 * A first point with x < 0 (before the note onset) is how the pre-onset slide-in (OpenUTAU snapFirst,
 * §3.2 layer ②) is encoded — the EDITOR sets that point's y to reference the previous note's tone.
 * ⚠ NOTE: this evaluator handles a point's contribution WITHIN one note's x-range; the cross-note-boundary
 * part of a pre-onset slide (rendering during the previous note's tail) is a Phase-5 refinement (§10.2) —
 * `evalF0Cents` will thread the neighbor context; verify against OpenUTAU before freezing golden vectors.
 */
export function evalPitchPoints(points: readonly PitchPoint[], x: number): number {
  const n = points.length;
  if (n === 0) return 0;
  const first = points[0]!;
  const last = points[n - 1]!;
  if (x <= first.x) return first.y; // hold before the first point
  if (x >= last.x) return last.y; // hold after the last point
  let i = 0;
  while (i < n - 1 && points[i + 1]!.x <= x) i++; // segment [i, i+1] with points[i].x ≤ x < points[i+1].x
  const a = points[i]!;
  const b = points[i + 1]!;
  const span = b.x - a.x;
  const t = span > 0 ? (x - a.x) / span : 1;
  return a.y + (b.y - a.y) * interpShape(t, b.shape); // the LATER point (b) owns the segment shape
}
