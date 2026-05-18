import { readFile } from "@tauri-apps/plugin-fs";

const TICKS_PER_BEAT = 480;

let audioCtx: AudioContext | null = null;
let activeSource: AudioBufferSourceNode | null = null;
let loadedBuffers: Map<string, AudioBuffer> = new Map();
let playStartCtxTime = 0;
let playStartOffsetSecs = 0;

function getContext(): AudioContext {
  if (!audioCtx || audioCtx.state === "closed") {
    audioCtx = new AudioContext();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

export async function loadAudioBuffer(filePath: string): Promise<AudioBuffer> {
  if (loadedBuffers.has(filePath)) {
    return loadedBuffers.get(filePath)!;
  }

  const ctx = getContext();
  const bytes = await readFile(filePath);
  const arrayBuffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength
  );
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

  loadedBuffers.set(filePath, audioBuffer);
  return audioBuffer;
}

export function playFromTick(
  filePath: string,
  startTick: number,
  tempo: number,
  onEnded?: () => void
): boolean {
  stopPlayback();

  const ctx = getContext();
  const buffer = loadedBuffers.get(filePath);
  if (!buffer) return false;

  const startSecs = ticksToSeconds(startTick, tempo);
  const offset = Math.max(0, Math.min(startSecs, buffer.duration));

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.onended = () => {
    activeSource = null;
    onEnded?.();
  };
  source.start(0, offset);
  activeSource = source;
  playStartCtxTime = ctx.currentTime;
  playStartOffsetSecs = offset;
  return true;
}

export function playFromTickWithOffset(
  filePath: string,
  playheadTick: number,
  tempo: number,
  segStartTick: number,
  segOffsetMs: number,
  onEnded?: () => void,
): boolean {
  stopPlayback();

  const ctx = getContext();
  const buffer = loadedBuffers.get(filePath);
  if (!buffer) return false;

  // How far into the segment is the playhead (in seconds)?
  const ticksIntoSeg = Math.max(0, playheadTick - segStartTick);
  const secsIntoSeg = ticksToSeconds(ticksIntoSeg, tempo);

  // Audio file offset = segment's trim offset + playhead position within segment
  const audioOffsetSecs = segOffsetMs / 1000.0 + secsIntoSeg;
  const offset = Math.max(0, Math.min(audioOffsetSecs, buffer.duration));

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.onended = () => {
    activeSource = null;
    onEnded?.();
  };
  source.start(0, offset);
  activeSource = source;
  playStartCtxTime = ctx.currentTime;
  playStartOffsetSecs = offset;
  return true;
}

export function stopPlayback() {
  if (activeSource) {
    try { activeSource.stop(); } catch { /* already stopped */ }
    activeSource = null;
  }
}

export function getCurrentTimeSecs(): number {
  if (!audioCtx || !activeSource) return playStartOffsetSecs;
  return playStartOffsetSecs + (audioCtx.currentTime - playStartCtxTime);
}

export function getContextTime(): number {
  return audioCtx?.currentTime ?? 0;
}

export function ticksToSeconds(ticks: number, tempo: number): number {
  return (ticks / TICKS_PER_BEAT) * (60.0 / tempo);
}

export function secondsToTicks(secs: number, tempo: number): number {
  return (secs * tempo / 60.0) * TICKS_PER_BEAT;
}

export function durationMsToTicks(ms: number, tempo: number): number {
  return secondsToTicks(ms / 1000.0, tempo);
}
