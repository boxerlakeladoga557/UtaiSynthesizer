import { useRef, useEffect, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import "./TimelineRuler.css";

const TICKS_PER_BEAT = 480;
const PIXELS_PER_TICK = 0.15;

interface Props {
  scrollX: number;
  zoom: number;
}

export function TimelineRuler({ scrollX, zoom }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { tempo, timeSignature } = useProjectStore();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const ppt = PIXELS_PER_TICK * zoom;
    const ticksPerBar = TICKS_PER_BEAT * timeSignature[0];
    const secsPerBar = (60.0 / tempo) * timeSignature[0];

    ctx.fillStyle = "#1a2236";
    ctx.fillRect(0, 0, width, height);

    const startTick = Math.floor(scrollX / ppt);
    const endTick = Math.ceil((scrollX + width) / ppt);
    const startBar = Math.floor(startTick / ticksPerBar);
    const endBar = Math.ceil(endTick / ticksPerBar);

    // Beat ticks
    for (let tick = startBar * ticksPerBar; tick < endTick; tick += TICKS_PER_BEAT) {
      const x = tick * ppt - scrollX;
      const isBar = tick % ticksPerBar === 0;

      if (isBar) {
        ctx.strokeStyle = "rgba(57, 197, 187, 0.4)";
        ctx.lineWidth = 1;
      } else {
        ctx.strokeStyle = "rgba(57, 197, 187, 0.15)";
        ctx.lineWidth = 0.5;
      }
      ctx.beginPath();
      ctx.moveTo(x, isBar ? 0 : height - 6);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Bar labels: number + timestamp
    for (let bar = startBar; bar <= endBar; bar++) {
      const tick = bar * ticksPerBar;
      const x = tick * ppt - scrollX;
      const barNum = bar + 1;
      const timeSecs = bar * secsPerBar;

      // Bar number
      ctx.fillStyle = "#e8ecf4";
      ctx.font = "bold 10px monospace";
      ctx.fillText(String(barNum), x + 3, 10);

      // Timestamp
      ctx.fillStyle = "#556b94";
      ctx.font = "9px monospace";
      ctx.fillText(formatTime(timeSecs), x + 3, 20);
    }

    // Bottom border
    ctx.strokeStyle = "#2a3a5c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 0.5);
    ctx.lineTo(width, height - 0.5);
    ctx.stroke();
  }, [scrollX, zoom, tempo, timeSignature]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  return <canvas ref={canvasRef} className="timeline-ruler" />;
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
