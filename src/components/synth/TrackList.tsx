import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useTranslation } from "react-i18next";
import type { Track } from "../../types/project";
import "./TrackList.css";

const TRACK_HEIGHT = 48;

interface Props {
  width: number;
  scrollY: number;
}

export function TrackList({ width, scrollY }: Props) {
  const { t } = useTranslation();
  const { tracks, updateTrack } = useProjectStore();
  const { activeTrackId, setActiveTrack } = useAppStore();

  return (
    <div className="track-list" style={{ width }}>
      <div className="track-list-scroll" style={{ transform: `translateY(${-scrollY}px)` }}>
        {tracks.length === 0 && (
          <div className="track-list-empty">
            <span className="text-muted">{t("tracks.empty")}</span>
          </div>
        )}
        {tracks.map((track) => (
          <TrackItem
            key={track.id}
            track={track}
            active={track.id === activeTrackId}
            onSelect={() => setActiveTrack(track.id)}
            onMute={() => updateTrack(track.id, { muted: !track.muted })}
            onSolo={() => updateTrack(track.id, { solo: !track.solo })}
          />
        ))}
      </div>
    </div>
  );
}

interface TrackItemProps {
  track: Track;
  active: boolean;
  onSelect: () => void;
  onMute: () => void;
  onSolo: () => void;
}

function TrackItem({ track, active, onSelect, onMute, onSolo }: TrackItemProps) {
  const { t } = useTranslation();
  const typeLabel = t(`tracks.${track.trackType}`);
  const colorVar =
    track.trackType === "vocal"
      ? "var(--track-vocal)"
      : track.trackType === "audio"
        ? "var(--track-audio)"
        : "var(--track-instrument)";

  return (
    <div
      className={`track-item ${active ? "active" : ""}`}
      style={{ height: TRACK_HEIGHT }}
      onClick={onSelect}
    >
      <div className="track-color-bar" style={{ background: colorVar }} />
      <div className="track-info">
        <span className="track-name">{track.name}</span>
        <span className="track-type text-muted">{typeLabel}</span>
      </div>
      <div className="track-controls">
        <button
          className={`track-btn ${track.muted ? "active-mute" : ""}`}
          onClick={(e) => { e.stopPropagation(); onMute(); }}
        >
          M
        </button>
        <button
          className={`track-btn ${track.solo ? "active-solo" : ""}`}
          onClick={(e) => { e.stopPropagation(); onSolo(); }}
        >
          S
        </button>
      </div>
    </div>
  );
}
