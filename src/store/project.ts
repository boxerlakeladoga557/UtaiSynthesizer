import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import type { Track } from "../types/project";

interface ProjectState {
  name: string;
  dirty: boolean;
  filePath: string | null;
  tracks: Track[];
  tempo: number;
  timeSignature: [number, number];
  selectedNotes: string[];
  playheadTick: number;

  newProject: (name: string) => Promise<void>;
  openProject: (path: string) => Promise<void>;
  saveProject: (path?: string) => Promise<void>;
  addTrack: (track: Track) => void;
  removeTrack: (id: string) => void;
  updateTrack: (id: string, updates: Partial<Track>) => void;
  setTempo: (bpm: number) => void;
  setPlayhead: (tick: number) => void;
  selectNotes: (ids: string[]) => void;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  name: "",
  dirty: false,
  filePath: null,
  tracks: [],
  tempo: 120,
  timeSignature: [4, 4],
  selectedNotes: [],
  playheadTick: 0,

  newProject: async (name) => {
    const result = await invoke<{ name: string; path: string | null }>(
      "new_project",
      { name }
    );
    set({
      name: result.name,
      filePath: result.path,
      dirty: false,
      tracks: [],
      tempo: 120,
      timeSignature: [4, 4],
    });
  },

  openProject: async (path) => {
    const result = await invoke<{
      name: string;
      path: string | null;
      tempo: number;
    }>("open_project", { path });
    set({
      name: result.name,
      filePath: result.path,
      dirty: false,
      tempo: result.tempo,
    });
  },

  saveProject: async (path) => {
    await invoke("save_project", { path: path ?? get().filePath });
    set({ dirty: false });
  },

  addTrack: (track) =>
    set((s) => ({ tracks: [...s.tracks, track], dirty: true })),

  removeTrack: (id) =>
    set((s) => ({
      tracks: s.tracks.filter((t) => t.id !== id),
      dirty: true,
    })),

  updateTrack: (id, updates) =>
    set((s) => ({
      tracks: s.tracks.map((t) => (t.id === id ? { ...t, ...updates } : t)),
      dirty: true,
    })),

  setTempo: (bpm) => {
    const TICKS_PER_BEAT = 480;
    set((s) => ({
      tempo: bpm,
      dirty: true,
      tracks: s.tracks.map((t) => ({
        ...t,
        segments: t.segments.map((seg) => {
          if (seg.content.type === "audioClip") {
            const ms = seg.content.totalDurationMs;
            const newTicks = Math.round((ms / 1000) * (bpm / 60) * TICKS_PER_BEAT);
            return { ...seg, durationTicks: newTicks };
          }
          return seg;
        }),
      })),
    }));
  },
  setPlayhead: (tick) => set({ playheadTick: tick }),
  selectNotes: (ids) => set({ selectedNotes: ids }),
}));
