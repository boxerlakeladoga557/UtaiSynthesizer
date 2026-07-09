// S50 Phase-5 foundation gate — the pure pitch evaluators (interpolateShape + evalF0Cents). Structural +
// exact-value tests; the COMPOSED golden-vector parity gate (overlay==Rust, incl. vibrato constants +
// cross-boundary pre-onset slide) lands in Phase 5 once the OpenUTAU formulas are verified (§10.2).
import { describe, it, expect } from "vitest";
import { interpShape, evalPitchPoints } from "./interpolateShape";
import { evalF0CentsAt, evalF0CentsFrames } from "./f0eval";
import type { Note } from "../types/project";

describe("interpolateShape", () => {
  it("easings hit boundaries + known midpoints, and clamp out-of-range t", () => {
    for (const s of ["linear", "sineIn", "sineOut", "sineInOut"] as const) {
      expect(interpShape(0, s)).toBeCloseTo(0, 6);
      expect(interpShape(1, s)).toBeCloseTo(1, 6);
    }
    expect(interpShape(0.5, "linear")).toBeCloseTo(0.5, 6);
    expect(interpShape(0.5, "sineIn")).toBeCloseTo(1 - Math.cos(Math.PI / 4), 6); // ≈0.2929
    expect(interpShape(0.5, "sineOut")).toBeCloseTo(Math.sin(Math.PI / 4), 6); // ≈0.7071
    expect(interpShape(0.5, "sineInOut")).toBeCloseTo(0.5, 6);
    expect(interpShape(-1, "sineIn")).toBe(0);
    expect(interpShape(2, "sineOut")).toBe(1);
  });

  it("evalPitchPoints holds outside the range + the LATER point owns the segment shape", () => {
    const lin = [{ x: 0, y: 0, shape: "linear" as const }, { x: 100, y: 200, shape: "linear" as const }];
    expect(evalPitchPoints(lin, -10)).toBe(0); // hold first
    expect(evalPitchPoints(lin, 110)).toBe(200); // hold last
    expect(evalPitchPoints(lin, 50)).toBeCloseTo(100, 6); // linear mid
    expect(evalPitchPoints([], 5)).toBe(0);
    const eased = [{ x: 0, y: 0, shape: "linear" as const }, { x: 100, y: 100, shape: "sineIn" as const }];
    expect(evalPitchPoints(eased, 50)).toBeCloseTo(100 * (1 - Math.cos(Math.PI / 4)), 6); // b (sineIn) owns it
  });
});

describe("evalF0Cents", () => {
  const mk = (id: string, tick: number, dur: number, pitch: number, extra: Partial<Note> = {}): Note =>
    ({ id, tick, duration: dur, pitch, lyric: "あ", velocity: 100, ...extra });
  const opts = { tempo: 120 };

  it("bare notes = stepped base (pitch*100 + detune); a gap tick = unvoiced", () => {
    const notes = [mk("a", 0, 480, 60), mk("b", 480, 480, 62, { detune: 20 })];
    expect(evalF0CentsAt(notes, undefined, 100, opts)).toEqual({ cents: 6000, voiced: true });
    expect(evalF0CentsAt(notes, undefined, 500, opts)).toEqual({ cents: 6220, voiced: true }); // 62*100+20
    expect(evalF0CentsAt(notes, undefined, 2000, opts)).toEqual({ cents: 0, voiced: false }); // past both = rest
  });

  it("pitchDev is additive; pitchPoints add on top of the base", () => {
    const notes = [mk("a", 0, 480, 60, { pitchPoints: [{ x: 0, y: 0, shape: "linear" }, { x: 480, y: 100, shape: "linear" }] })];
    const dev = { xs: [0, 480], ys: [0, 50] };
    const r = evalF0CentsAt(notes, dev, 240, opts); // base 6000 + pitchPoints(50) + pitchDev(25)
    expect(r.voiced).toBe(true);
    expect(r.cents).toBeCloseTo(6000 + 50 + 25, 4);
  });

  it("vibrato is 0 before the tail span, nonzero inside, bounded by depth", () => {
    const notes = [mk("a", 0, 960, 60, { vibrato: { length: 0.5, period: 200, depth: 50, in: 0, out: 0, shift: 0, drift: 0 } })];
    expect(evalF0CentsAt(notes, undefined, 240, opts).cents).toBe(6000); // span = last 0.5 → starts at 480; before = base only
    const inside = evalF0CentsAt(notes, undefined, 720, opts).cents;
    expect(Math.abs(inside - 6000)).toBeLessThanOrEqual(50 + 1e-6); // within ±depth
  });

  it("evalF0CentsFrames yields the per-frame cents array + voiced mask", () => {
    const notes = [mk("a", 0, 100, 60)];
    const { cents, voiced } = evalF0CentsFrames(notes, undefined, { frameStartTick: 0, ticksPerFrame: 50, frameCount: 4 }, opts);
    expect(cents.length).toBe(4);
    expect(Array.from(voiced)).toEqual([1, 1, 0, 0]); // ticks 0,50 ∈ [0,100); 100,150 rest
    expect(cents[0]).toBe(6000);
  });
});
