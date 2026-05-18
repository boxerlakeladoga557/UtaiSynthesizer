import { useRef, useCallback } from "react";
import { Toolbar } from "./Toolbar";
import { TrackList } from "./TrackList";
import { TimelineRuler } from "./TimelineRuler";
import { Arrangement } from "./Arrangement";
import { HScrollbar } from "./HScrollbar";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import "./DawView.css";

const HEADER_WIDTH = 200;
const PIXELS_PER_TICK = 0.15;
const TICKS_PER_BEAT = 480;

export function DawView() {
  const { scrollX, scrollY, setScroll, zoom, setZoom } = useAppStore();
  const { tracks, timeSignature } = useProjectStore();
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  const totalTicks = Math.max(
    TICKS_PER_BEAT * timeSignature[0] * 32,
    ...tracks.flatMap((t) => t.segments.map((s) => s.startTick + s.durationTicks)),
  );
  const totalWidth = totalTicks * PIXELS_PER_TICK * zoom;

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (e.ctrlKey) {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        setZoom(zoom * factor);
      } else if (e.shiftKey) {
        setScroll(Math.max(0, scrollX + e.deltaY), scrollY);
      } else {
        setScroll(scrollX, Math.max(0, scrollY + e.deltaY));
      }
    },
    [scrollX, scrollY, zoom, setScroll, setZoom]
  );

  return (
    <div className="daw-view">
      <Toolbar />
      <div className="daw-grid" onWheel={handleWheel}>
        <div className="daw-corner" style={{ width: HEADER_WIDTH }} />
        <TimelineRuler scrollX={scrollX} zoom={zoom} />
        <TrackList width={HEADER_WIDTH} scrollY={scrollY} />
        <div className="daw-canvas-container" ref={canvasContainerRef}>
          <Arrangement />
        </div>
        <div className="daw-scrollbar-corner" style={{ width: HEADER_WIDTH }} />
        <HScrollbar
          scrollX={scrollX}
          totalWidth={totalWidth}
          viewWidth={800}
          onChange={(x) => setScroll(x, scrollY)}
        />
      </div>
    </div>
  );
}
