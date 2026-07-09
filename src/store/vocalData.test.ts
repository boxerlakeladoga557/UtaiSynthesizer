// S48 Phase 3 GATE — vocal DATA MODEL / store / undo / .usp, driven headless (no editor UI).
// Verifies the three Phase-3 claims: (A) undo captures a vocal-note edit incl. a NEW field, (B) it
// reverts cleanly to the saved baseline (dirty recomputed), (C) .usp save/load round-trips every vocal
// field byte-identically, (D) no false-dirty (normalizeNote strips default optionals; determinism).
import { describe, it, expect, beforeEach, vi } from "vitest";

// The store's fire-and-forget backend log (history dbg → logToBackend → invoke) + any cancel_voice must
// not throw in a headless run; i18n (announce banner) is mocked so react-i18next/JSON never load.
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../i18n", () => ({ default: { t: (k: string) => k } }));

import { useProjectStore } from "./project";
import { useHistoryStore, installHistory } from "./history";
import { useAppStore } from "./app";
import { buildSaveBundle, buildAutosaveJson, parseLoadedBundle } from "../lib/project/bundle";
import type { Track, Segment, SegmentContent, Note, VocalTrackParams } from "../types/project";

type NotesContent = Extract<SegmentContent, { type: "notes" }>;

const T = "t1";
const S = "seg1";

function plainNote(): Note {
  return { id: "n1", tick: 0, duration: 240, pitch: 60, lyric: "か", velocity: 100 };
}
function notesSeg(notes: Note[], extra: Partial<NotesContent> = {}): Segment {
  return { id: S, startTick: 0, durationTicks: 1920, content: { type: "notes", notes, ...extra } };
}
function vocalTrack(seg: Segment, params?: VocalTrackParams): Track {
  return {
    id: T, name: "Vocal", trackType: "vocal", segments: [seg],
    volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {},
    ...(params ? { vocalParams: params } : {}),
  };
}
function seed(track: Track) {
  useProjectStore.setState({
    name: "P", tracks: [track], tempo: 120, timeSignature: [4, 4],
    dirty: false, filePath: null, selectedNotes: [], playheadTick: 0,
  });
}
function notes(): Note[] {
  return (useProjectStore.getState().tracks[0]!.segments[0]!.content as NotesContent).notes;
}
function content(): NotesContent {
  return useProjectStore.getState().tracks[0]!.segments[0]!.content as NotesContent;
}

let uninstall: (() => void) | null = null;
beforeEach(() => {
  useAppStore.setState({ selectedSegment: null, selectedSegments: [], activeTrackId: null } as never);
  seed(vocalTrack(notesSeg([plainNote()])));
  uninstall?.();
  uninstall = installHistory();
  useHistoryStore.getState().reset();
  useHistoryStore.getState().markSaved(); // baseline = the seeded doc
});

describe("Phase 3 — undo captures + reverts vocal edits (GATE A/B)", () => {
  it("captures an edit to a NEW field (detune) and reverts it clean", () => {
    expect(useProjectStore.getState().dirty).toBe(false);
    useProjectStore.getState().updateVocalNote(T, S, "n1", { detune: 30 });
    expect(notes()[0]!.detune).toBe(30);
    expect(useHistoryStore.getState().canUndo).toBe(true); // the new field was in contentSig → captured
    expect(useProjectStore.getState().dirty).toBe(true);

    useHistoryStore.getState().undo();
    expect(notes()[0]!.detune).toBeUndefined(); // reverted to the seeded (no-detune) note
    expect(useProjectStore.getState().dirty).toBe(false); // sig back to savedSig
    expect(useHistoryStore.getState().canRedo).toBe(true);
  });

  it("captures add / delete note and vocalParams as distinct undo steps", () => {
    useProjectStore.getState().addVocalNote(T, S, { ...plainNote(), id: "n2", tick: 480, pitch: 62 });
    expect(notes()).toHaveLength(2);
    useProjectStore.getState().setVocalParams(T, { transpose: 3 });
    expect(useProjectStore.getState().tracks[0]!.vocalParams).toMatchObject({ transpose: 3, backend: "sovits" });

    useHistoryStore.getState().undo(); // undo setVocalParams
    expect(useProjectStore.getState().tracks[0]!.vocalParams?.transpose ?? 0).toBe(0);
    expect(notes()).toHaveLength(2);
    useHistoryStore.getState().undo(); // undo addVocalNote
    expect(notes()).toHaveLength(1);
    expect(useProjectStore.getState().dirty).toBe(false);
  });

  it("captures pitch curves (pitchDev / paramCurves / pitchPoints)", () => {
    useProjectStore.getState().setSegmentPitchDev(T, S, { xs: [0, 240], ys: [0, 50] });
    expect(content().pitchDev).toEqual({ xs: [0, 240], ys: [0, 50] });
    expect(useHistoryStore.getState().canUndo).toBe(true);
    useHistoryStore.getState().undo();
    expect(content().pitchDev).toBeUndefined();
    expect(useProjectStore.getState().dirty).toBe(false);
  });
});

describe("Phase 3 — .usp save/load round-trips every vocal field (GATE C)", () => {
  const rich = vocalTrack(
    notesSeg(
      [
        {
          id: "n1", tick: 0, duration: 240, pitch: 60, lyric: "か", velocity: 100,
          detune: 30, tie: true, pitchAuto: false, lang: "ja", phonemeInput: "ka",
          pitchPoints: [
            { x: -20, y: -100, shape: "sineIn" },
            { x: 120, y: 0, shape: "linear" },
          ],
          vibrato: { length: 0.5, period: 200, depth: 50, in: 0.1, out: 0.1, shift: 0, drift: 0 },
        },
      ],
      { pitchDev: { xs: [0, 240], ys: [0, 50] }, paramCurves: { loudness: { xs: [0, 480], ys: [0, -3] } } },
    ),
    { backend: "sovits", speakerId: 49, langId: 2, transpose: 2 },
  );

  it("preserves vocalParams + all note/curve fields through save→load", () => {
    const { projectJson } = buildSaveBundle("P", [rich], 120, [4, 4]);
    const loaded = parseLoadedBundle(projectJson, "C:/proj.usp");
    expect(loaded.tracks[0]!.vocalParams).toEqual(rich.vocalParams);
    expect(loaded.tracks[0]!.segments[0]!.content).toEqual(rich.segments[0]!.content);
  });

  it("load→serialize is byte-identical (autosave form)", () => {
    const auto1 = buildAutosaveJson("P", [rich], 120, [4, 4]);
    const reloaded = parseLoadedBundle(auto1, "C:/proj.usp");
    const auto2 = buildAutosaveJson("P", reloaded.tracks, reloaded.tempo, reloaded.timeSignature);
    expect(auto2).toBe(auto1);
  });
});

describe("Phase 3 — no false-dirty (GATE D)", () => {
  it("serializing the same doc twice is byte-identical", () => {
    const t = vocalTrack(notesSeg([plainNote()]));
    expect(buildAutosaveJson("P", [t], 120, [4, 4])).toBe(buildAutosaveJson("P", [t], 120, [4, 4]));
  });

  it("normalizeNote strips default optionals on write (no JSON growth)", () => {
    // A note added with explicit default values must NOT store them (detune:0 / tie:false → absent).
    useProjectStore.getState().addVocalNote(T, S, {
      ...plainNote(), id: "n2", tick: 480, pitch: 62, detune: 0, tie: false, pitchAuto: true, pitchPoints: [],
    });
    const n2 = notes().find((n) => n.id === "n2")!;
    expect(n2.detune).toBeUndefined();
    expect(n2.tie).toBeUndefined();
    expect(n2.pitchAuto).toBeUndefined();
    expect(n2.pitchPoints).toBeUndefined();
  });

  it("setting a field back to its default returns to the byte-identical baseline", () => {
    const base = buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4]);
    useProjectStore.getState().updateVocalNote(T, S, "n1", { detune: 15 });
    expect(buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4])).not.toBe(base);
    useProjectStore.getState().updateVocalNote(T, S, "n1", { detune: 0 }); // back to default
    expect(buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4])).toBe(base);
  });

  it("paramCurves key order is canonical — delete-then-readd does not false-dirty", () => {
    const P = () => useProjectStore.getState();
    P().setSegmentParamCurve(T, S, "loudness", { xs: [0], ys: [0] });
    P().setSegmentParamCurve(T, S, "tension", { xs: [0], ys: [0] });
    const baseline = buildAutosaveJson("P", P().tracks, 120, [4, 4]);
    // delete then re-add loudness → without sorted keys the Record would reorder to {tension, loudness}
    P().setSegmentParamCurve(T, S, "loudness", undefined);
    P().setSegmentParamCurve(T, S, "loudness", { xs: [0], ys: [0] });
    expect(buildAutosaveJson("P", P().tracks, 120, [4, 4])).toBe(baseline);
    expect(Object.keys(content().paramCurves!)).toEqual(["loudness", "tension"]); // sorted
  });

  it("normalizeNote canonicalizes vibrato/pitchPoints element key order (input order can't false-dirty)", () => {
    const P = () => useProjectStore.getState();
    // same values, NON-canonical key/element order (a future editor might build objects either way)
    P().updateVocalNote(T, S, "n1", {
      vibrato: { drift: 0, shift: 0, out: 0.1, in: 0.1, depth: 50, period: 200, length: 0.5 },
      pitchPoints: [{ shape: "linear", y: 0, x: 120 }, { y: -100, shape: "sineIn", x: -20 }],
    });
    const jsonA = buildAutosaveJson("P", P().tracks, 120, [4, 4]);
    // same values, canonical key/element order
    P().updateVocalNote(T, S, "n1", {
      vibrato: { length: 0.5, period: 200, depth: 50, in: 0.1, out: 0.1, shift: 0, drift: 0 },
      pitchPoints: [{ x: -20, y: -100, shape: "sineIn" }, { x: 120, y: 0, shape: "linear" }],
    });
    expect(buildAutosaveJson("P", P().tracks, 120, [4, 4])).toBe(jsonA); // normalized → identical bytes
  });
});
