import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

export interface AudioTrackData {
  filePath: string;
  durationMs: number;
  sampleRate: number;
  peaks: number[];
}

interface AudioState {
  trackAudio: Record<string, AudioTrackData>;
  isPlaying: boolean;
  playStartTime: number;
  playStartTick: number;

  loadAudioFile: (trackId: string, filePath: string) => Promise<AudioTrackData>;
  setPlaying: (playing: boolean) => void;
  setPlayStart: (time: number, tick: number) => void;
}

export const useAudioStore = create<AudioState>((set) => ({
  trackAudio: {},
  isPlaying: false,
  playStartTime: 0,
  playStartTick: 0,

  loadAudioFile: async (trackId, filePath) => {
    const info = await invoke<{
      duration_ms: number;
      sample_rate: number;
      channels: number;
      peaks: number[];
    }>("load_audio_file", { path: filePath });

    const data: AudioTrackData = {
      filePath,
      durationMs: info.duration_ms,
      sampleRate: info.sample_rate,
      peaks: info.peaks,
    };

    set((s) => ({
      trackAudio: { ...s.trackAudio, [trackId]: data },
    }));

    return data;
  },

  setPlaying: (playing) => set({ isPlaying: playing }),
  setPlayStart: (time, tick) => set({ playStartTime: time, playStartTick: tick }),
}));
