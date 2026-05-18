import { useRef, useEffect, useCallback, useState } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import "./Arrangement.css";

const TRACK_HEIGHT = 48;
const PIXELS_PER_TICK = 0.15;
const TICKS_PER_BEAT = 480;
const EDGE_ZONE = 6;

type DragMode = null | "playhead" | "move" | "resizeL" | "resizeR";

interface DragState {
  mode: DragMode;
  trackIdx: number;
  segId: string;
  startMouseX: number;
  origStartTick: number;
  origDurationTicks: number;
  origOffsetMs: number;
}

export function Arrangement() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { tracks, timeSignature, playheadTick, setPlayhead, updateTrack, tempo } = useProjectStore();
  const { zoom, scrollX, scrollY, openWorkflow } = useAppStore();
  const { trackAudio } = useAudioStore();
  const [cursor, setCursor] = useState("default");
  const dragRef = useRef<DragState | null>(null);
  const mouseXRef = useRef(-9999);
  const drawRef = useRef(() => {});

  const ppt = PIXELS_PER_TICK * zoom;

  const canvasToTick = useCallback((clientX: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    return Math.max(0, Math.round((clientX - rect.left + scrollX) / ppt));
  }, [scrollX, ppt]);

  const hitTest = useCallback((clientX: number, clientY: number): { trackIdx: number; segId: string; zone: "body" | "left" | "right" } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left + scrollX;
    const y = clientY - rect.top + scrollY;

    for (let i = 0; i < tracks.length; i++) {
      const track = tracks[i];
      if (!track) continue;
      const trackY = i * TRACK_HEIGHT;
      if (y < trackY || y > trackY + TRACK_HEIGHT) continue;

      for (const seg of track.segments) {
        const sx = seg.startTick * ppt;
        const sw = seg.durationTicks * ppt;
        if (x >= sx && x <= sx + sw) {
          if (x - sx < EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "left" };
          if (sx + sw - x < EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "right" };
          return { trackIdx: i, segId: seg.id, zone: "body" };
        }
      }
    }
    return null;
  }, [tracks, ppt, scrollX, scrollY]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const hit = hitTest(e.clientX, e.clientY);
    if (hit) {
      const track = tracks[hit.trackIdx];
      const seg = track?.segments.find((s) => s.id === hit.segId);
      if (!track || !seg) return;

      const mode: DragMode = hit.zone === "left" ? "resizeL" : hit.zone === "right" ? "resizeR" : "move";
      const offsetMs = seg.content.type === "audioClip" ? seg.content.offsetMs : 0;
      dragRef.current = {
        mode, trackIdx: hit.trackIdx, segId: hit.segId,
        startMouseX: e.clientX, origStartTick: seg.startTick,
        origDurationTicks: seg.durationTicks, origOffsetMs: offsetMs,
      };
    } else {
      dragRef.current = { mode: "playhead", trackIdx: -1, segId: "", startMouseX: e.clientX, origStartTick: 0, origDurationTicks: 0, origOffsetMs: 0 };
      setPlayhead(canvasToTick(e.clientX));
    }
  }, [hitTest, tracks, setPlayhead, canvasToTick]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const drag = dragRef.current;

    const canvas = canvasRef.current;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      mouseXRef.current = e.clientX - rect.left;
    }

    if (!drag) {
      const hit = hitTest(e.clientX, e.clientY);
      if (!hit) { setCursor("crosshair"); } else {
        setCursor(hit.zone === "left" || hit.zone === "right" ? "ew-resize" : "grab");
      }
      drawRef.current();
      return;
    }

    if (drag.mode === "playhead") {
      setPlayhead(canvasToTick(e.clientX));
      return;
    }

    const deltaPx = e.clientX - drag.startMouseX;
    const deltaTicks = Math.round(deltaPx / ppt);
    const track = tracks[drag.trackIdx];
    if (!track) return;

    const updated = track.segments.map((seg) => {
      if (seg.id !== drag.segId) return seg;

      if (drag.mode === "move") {
        return { ...seg, startTick: Math.max(0, drag.origStartTick + deltaTicks) };
      }
      if (drag.mode === "resizeL") {
        const newStart = Math.max(0, drag.origStartTick + deltaTicks);
        const shrink = newStart - drag.origStartTick;
        const newDuration = Math.max(TICKS_PER_BEAT / 4, drag.origDurationTicks - shrink);
        const newOffsetMs = seg.content.type === "audioClip"
          ? Math.max(0, drag.origOffsetMs + (shrink / TICKS_PER_BEAT) * (60000 / tempo))
          : 0;
        return {
          ...seg, startTick: newStart, durationTicks: newDuration,
          content: seg.content.type === "audioClip" ? { ...seg.content, offsetMs: newOffsetMs } : seg.content,
        };
      }
      if (drag.mode === "resizeR") {
        return { ...seg, durationTicks: Math.max(TICKS_PER_BEAT / 4, drag.origDurationTicks + deltaTicks) };
      }
      return seg;
    });

    updateTrack(track.id, { segments: updated });
  }, [hitTest, tracks, ppt, tempo, setPlayhead, canvasToTick, updateTrack]);

  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const handleMouseLeave = useCallback(() => { mouseXRef.current = -9999; dragRef.current = null; drawRef.current(); }, []);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    const hit = hitTest(e.clientX, e.clientY);
    if (hit) openWorkflow(hit.segId);
  }, [hitTest, openWorkflow]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const ticksPerBar = TICKS_PER_BEAT * timeSignature[0];

    ctx.fillStyle = "#0d1220";
    ctx.fillRect(0, 0, width, height);

    const startTick = Math.floor(scrollX / ppt);
    const endTick = Math.ceil((scrollX + width) / ppt);
    for (let tick = startTick - (startTick % TICKS_PER_BEAT); tick < endTick; tick += TICKS_PER_BEAT) {
      const x = tick * ppt - scrollX;
      const isBar = tick % ticksPerBar === 0;
      ctx.strokeStyle = isBar ? "rgba(57,197,187,0.18)" : "rgba(57,197,187,0.05)";
      ctx.lineWidth = isBar ? 1 : 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    for (let i = 0; i < tracks.length; i++) {
      const track = tracks[i];
      if (!track) continue;
      const y = i * TRACK_HEIGHT - scrollY;

      ctx.strokeStyle = "rgba(30,42,69,0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y + TRACK_HEIGHT);
      ctx.lineTo(width, y + TRACK_HEIGHT);
      ctx.stroke();

      const audio = trackAudio[track.id];
      const c = track.trackType === "audio" ? [96, 165, 250] : track.trackType === "vocal" ? [57, 197, 187] : [167, 139, 250];

      for (const seg of track.segments) {
        const sx = seg.startTick * ppt - scrollX;
        const sw = seg.durationTicks * ppt;
        const sy = y + 2;
        const sh = TRACK_HEIGHT - 4;

        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.15)`;
        ctx.fillRect(sx, sy, sw, sh);
        ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},0.5)`;
        ctx.lineWidth = 1;
        ctx.strokeRect(sx, sy, sw, sh);

        // Edge handles
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.3)`;
        ctx.fillRect(sx, sy, 3, sh);
        ctx.fillRect(sx + sw - 3, sy, 3, sh);

        if (audio && audio.peaks.length > 0 && sw > 2) {
          const offMs = seg.content.type === "audioClip" ? seg.content.offsetMs : 0;
          const totalMs = seg.content.type === "audioClip" ? seg.content.totalDurationMs : audio.durationMs;
          drawWaveform(ctx, audio.peaks, sx, sy, sw, sh, `rgba(${c[0]},${c[1]},${c[2]},0.6)`, offMs, totalMs);
        }
      }
    }

    const phx = playheadTick * ppt - scrollX;
    if (phx >= -1 && phx <= width + 1) {
      const nearPlayhead = Math.abs(mouseXRef.current - phx) < 10;

      if (nearPlayhead) {
        ctx.save();
        ctx.shadowColor = "#ff6b9d";
        ctx.shadowBlur = 12;
        ctx.strokeStyle = "#ffadc8";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(phx, 0);
        ctx.lineTo(phx, height);
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.strokeStyle = "#ff6b9d";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(phx, 0);
        ctx.lineTo(phx, height);
        ctx.stroke();
      }
      ctx.fillStyle = nearPlayhead ? "#ffadc8" : "#ff6b9d";
      ctx.beginPath();
      ctx.moveTo(phx - 6, 0);
      ctx.lineTo(phx + 6, 0);
      ctx.lineTo(phx, 8);
      ctx.closePath();
      ctx.fill();
    }
  }, [tracks, trackAudio, ppt, scrollX, scrollY, playheadTick, timeSignature]);

  drawRef.current = draw;

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  return (
    <div className="arrangement">
      <canvas
        ref={canvasRef}
        className="arrangement-canvas"
        style={{ cursor }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
      />
    </div>
  );
}

function drawWaveform(
  ctx: CanvasRenderingContext2D,
  peaks: number[],
  x: number, y: number, w: number, h: number,
  color: string,
  offsetMs: number,
  totalDurationMs: number,
) {
  const midY = y + h / 2;
  const amp = h / 2 - 1;
  const startRatio = totalDurationMs > 0 ? offsetMs / totalDurationMs : 0;

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let px = 0; px < w; px++) {
    const ratio = startRatio + (px / w) * (1 - startRatio);
    const peakIdx = Math.min(Math.floor(ratio * peaks.length), peaks.length - 1);
    if (peakIdx < 0) continue;
    const peak = peaks[peakIdx] ?? 0;
    ctx.moveTo(x + px, midY - peak * amp);
    ctx.lineTo(x + px, midY + peak * amp);
  }
  ctx.stroke();
}
