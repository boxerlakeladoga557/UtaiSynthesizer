// ② Vocal render (S48 Phase 6) — buildVocalScore alignment gate. The score triples' `frames` and the
// Option-A f0 array MUST share one 50fps grid so Σ(triple frames) == f0 length (build_note_hz maps cv↔DAW
// by cumulative frames — a length disagreement silently drifts pitch, the class the user has been burned by).
import { describe, it, expect, vi } from "vitest";

// buildVocalScore is pure, but the module also imports invoke/store (for renderVocalSegment) — mock so the
// module loads headless (mirrors store/vocalData.test.ts).
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import { buildVocalScore } from "./vocalRender";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note } from "../../types/project";

const mkNote = (id: string, tick: number, duration: number, pitch: number, lyric = "あ"): Note => ({
  id, tick, duration, pitch, lyric, velocity: 100,
});

describe("buildVocalScore", () => {
  const tempo = 120;
  const def = DEFAULT_TRANSITION;

  it("aligns Σ(triple frames) == f0 length == voiced length (build_note_hz cv↔DAW invariant)", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 960, 480, 62)]; // gap 480..960
    const { triples, f0Cents, f0Voiced } = buildVocalScore(notes, undefined, tempo, def, "AP");
    const sum = triples.reduce((s, t) => s + t.frames, 0);
    expect(f0Cents.length).toBe(sum);
    expect(f0Voiced.length).toBe(sum);
    expect(f0Cents.length).toBeGreaterThan(0);
  });

  it("inserts a leading rest + explicit gap rests (§3.4 — never inferred from pitch==0)", () => {
    const notes = [mkNote("a", 480, 480, 60), mkNote("b", 1440, 480, 62)]; // starts at 480; gap 960..1440
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples[0]!.lyric).toBe("R"); // leading rest so stem-ms 0 == segment start
    expect(triples[0]!.note_num).toBe(0);
    expect(triples.filter((t) => t.lyric === "R").length).toBe(2); // leading + inter-note gap
    expect(triples.filter((t) => t.lyric !== "R").map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("abutting notes glide with NO rest between", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 480, 480, 62)]; // abut at 480
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples.filter((t) => t.lyric === "R").length).toBe(0);
    expect(triples.map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("passes RAW pitch (transpose is applied Rust-side, §9.3)", () => {
    const notes = [mkNote("a", 0, 480, 60)];
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples.find((t) => t.lyric !== "R")!.note_num).toBe(60);
  });

  it("keeps each note's lyric (JA kana), sorts by tick", () => {
    const notes = [mkNote("b", 480, 480, 62, "き"), mkNote("a", 0, 480, 60, "か")]; // unsorted input
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples.map((t) => t.lyric)).toEqual(["か", "き"]);
  });

  it("empty notes → empty score + empty f0", () => {
    const { triples, f0Cents } = buildVocalScore([], undefined, tempo, def, "AP");
    expect(triples.length).toBe(0);
    expect(f0Cents.length).toBe(0);
  });

  // helper: the [start,end) frame span of each triple.
  const spans = (triples: { lyric: string; frames: number }[]) => {
    let c = 0;
    return triples.map((t) => { const s = c; c += t.frames; return { lyric: t.lyric, s, e: c }; });
  };

  it("breath note → AP phone + UNVOICED f0 (breaks the pitch chain, §M3)", () => {
    // か—AP—き, all abutting. The AP breath is emitted as the AP phone and its frames are UNVOICED (so the
    // か releases / the き scoops rather than gliding into/out of the breath).
    const notes = [mkNote("a", 0, 480, 60, "か"), mkNote("br", 480, 240, 62, "AP"), mkNote("c", 720, 480, 64, "き")];
    const { triples, f0Voiced } = buildVocalScore(notes, undefined, tempo, def, "AP");
    const ap = spans(triples).find((x) => x.lyric === "AP")!;
    expect(ap).toBeTruthy(); // breath kept as the AP phone (not silence, not "か")
    for (let f = ap.s; f < ap.e; f++) expect(f0Voiced[f]).toBe(0); // breath frames unvoiced
    expect(Array.from(f0Voiced).some((v) => v === 1)).toBe(true); // the sung notes are voiced
  });

  it("custom breath token is unvoiced; renaming it re-voices the OLD token (§user dynamic)", () => {
    const notes = [mkNote("a", 0, 480, 60, "呼")];
    // breathToken "呼" → the note IS a breath → AP phone, all-unvoiced.
    const asBreath = buildVocalScore(notes, undefined, tempo, def, "呼");
    expect(asBreath.triples.some((t) => t.lyric === "AP")).toBe(true);
    expect(Array.from(asBreath.f0Voiced).every((v) => v === 0)).toBe(true);
    // change the token away → "呼" is a normal lyric again → sent literally + VOICED (connected pitch).
    const asLyric = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(asLyric.triples.some((t) => t.lyric === "呼")).toBe(true);
    expect(Array.from(asLyric.f0Voiced).some((v) => v === 1)).toBe(true);
  });
});
