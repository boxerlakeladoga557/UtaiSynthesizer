import { useState, useEffect, useRef } from "react";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import * as playback from "../../lib/audio/playback";
import { OverviewMap } from "./OverviewMap";
import i18n from "../../i18n";
import type { Track, Segment } from "../../types/project";
import "./Toolbar.css";

export function Toolbar() {
  const { t } = useTranslation();
  const { addTrack, tempo, setTempo, playheadTick, setPlayhead, timeSignature, tracks } =
    useProjectStore();
  const { loadAudioFile, trackAudio, isPlaying, setPlaying, setPlayStart } = useAudioStore();
  const [showAddMenu, setShowAddMenu] = useState(false);
  const animRef = useRef<number>(0);

  // Playhead animation during playback
  useEffect(() => {
    if (!isPlaying) {
      cancelAnimationFrame(animRef.current);
      return;
    }

    const startTime = playback.getContextTime();
    const startTick = playheadTick;

    const animate = () => {
      const elapsed = playback.getContextTime() - startTime;
      const ticksElapsed = playback.secondsToTicks(elapsed, tempo);
      setPlayhead(Math.round(startTick + ticksElapsed));
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animRef.current);
  }, [isPlaying, tempo, playheadTick, setPlayhead]);

  const handlePlay = () => {
    if (isPlaying) return;

    for (const track of tracks) {
      const data = trackAudio[track.id];
      if (!data) continue;

      // Find the segment the playhead is in (or the first one)
      const seg = track.segments.find(
        (s) => playheadTick >= s.startTick && playheadTick < s.startTick + s.durationTicks
      ) ?? track.segments[0];
      if (!seg) continue;

      const segOffsetMs = seg.content.type === "audioClip" ? seg.content.offsetMs : 0;

      playback.loadAudioBuffer(data.filePath).then(() => {
        playback.playFromTickWithOffset(
          data.filePath, playheadTick, tempo,
          seg.startTick, segOffsetMs,
          () => setPlaying(false),
        );
        setPlaying(true);
        setPlayStart(playback.getContextTime(), playheadTick);
      });
      return;
    }
  };

  const handleStop = () => {
    playback.stopPlayback();
    setPlaying(false);
    setPlayhead(0);
  };

  const handleImportAudio = async () => {
    setShowAddMenu(false);
    try {
      const path = await open({
        title: t("toolbar.importAudio"),
        filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "ogg"] }],
      });
      if (!path) return;

      const filePath = path as string;
      const fileName = filePath.split(/[/\\]/).pop() ?? "audio";
      const trackId = `track-${Date.now()}`;

      // Load audio through Rust backend → get real duration + peaks
      const audioData = await loadAudioFile(trackId, filePath);
      const durationTicks = playback.durationMsToTicks(audioData.durationMs, tempo);

      const seg: Segment = {
        id: `seg-${Date.now()}`,
        startTick: 0,
        durationTicks: Math.round(durationTicks),
        content: { type: "audioClip", sourcePath: filePath, offsetMs: 0, totalDurationMs: audioData.durationMs },
      };
      const track: Track = {
        id: trackId,
        name: fileName,
        trackType: "audio",
        segments: [seg],
        volumeDb: 0,
        pan: 0,
        muted: false,
        solo: false,
      };
      addTrack(track);

      // Pre-load into Web Audio
      playback.loadAudioBuffer(filePath);
    } catch (e) {
      console.error("Import failed:", e);
    }
  };

  const handleAddMidiTrack = () => {
    setShowAddMenu(false);
    const track: Track = {
      id: `track-${Date.now()}`,
      name: `MIDI ${Math.floor(Math.random() * 100)}`,
      trackType: "vocal",
      segments: [],
      volumeDb: 0,
      pan: 0,
      muted: false,
      solo: false,
    };
    addTrack(track);
  };

  const cycleLang = () => {
    const langs = ["zh", "en", "ja"];
    const cur = langs.indexOf(i18n.language);
    const next = langs[(cur + 1) % langs.length]!;
    i18n.changeLanguage(next);
  };

  return (
    <div className="toolbar">
      <div className="toolbar-section transport">
        <button className="transport-btn" onClick={handleStop} data-tooltip={t("transport.stop")}>
          &#x23F9;
        </button>
        <button
          className={`transport-btn play ${isPlaying ? "playing" : ""}`}
          onClick={handlePlay}
          data-tooltip={t("transport.play")}
        >
          &#x25B6;
        </button>
        <button className="transport-btn rec" data-tooltip={t("transport.record")}>
          &#x23FA;
        </button>
      </div>

      <OverviewMap />

      <div className="toolbar-divider" />

      <div className="toolbar-section tempo-section">
        <label className="toolbar-label">{t("toolbar.bpm")}</label>
        <input
          type="number"
          className="tempo-input mono"
          value={tempo}
          min={20}
          max={400}
          step={1}
          onChange={(e) => setTempo(Number(e.target.value))}
        />
      </div>

      <div className="toolbar-section time-sig">
        <span className="mono time-display">
          {timeSignature[0]}/{timeSignature[1]}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section position-section">
        <label className="toolbar-label">{t("toolbar.position")}</label>
        <span className="mono position-display">
          {formatPosition(playheadTick, timeSignature)}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section snap-section">
        <label className="toolbar-label">{t("toolbar.snap")}</label>
        <select className="snap-select" defaultValue="16">
          <option value="4">1/4</option>
          <option value="8">1/8</option>
          <option value="16">1/16</option>
          <option value="32">1/32</option>
          <option value="triplet">{t("toolbar.triplet")}</option>
          <option value="free">{t("toolbar.free")}</option>
        </select>
      </div>

      <div className="toolbar-spacer" />

      <div className="toolbar-section" style={{ position: "relative" }}>
        <button className="toolbar-btn" onClick={() => setShowAddMenu(!showAddMenu)}>
          + {t("toolbar.addTrack")}
        </button>
        {showAddMenu && (
          <div className="add-track-menu">
            <button className="add-track-option" onClick={handleImportAudio}>
              {t("toolbar.importAudio")}
            </button>
            <button className="add-track-option" onClick={handleAddMidiTrack}>
              {t("toolbar.addMidi")}
            </button>
          </div>
        )}
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section">
        <button className="toolbar-btn lang-btn mono" onClick={cycleLang}>
          {i18n.language.toUpperCase()}
        </button>
      </div>
    </div>
  );
}

function formatPosition(tick: number, timeSig: [number, number]): string {
  const ticksPerBeat = 480;
  const ticksPerBar = ticksPerBeat * timeSig[0];
  const bar = Math.floor(tick / ticksPerBar) + 1;
  const beat = Math.floor((tick % ticksPerBar) / ticksPerBeat) + 1;
  const sub = Math.floor(((tick % ticksPerBar) % ticksPerBeat) / (ticksPerBeat / 4));
  return `${bar}:${beat}:${sub.toString().padStart(2, "0")}`;
}
