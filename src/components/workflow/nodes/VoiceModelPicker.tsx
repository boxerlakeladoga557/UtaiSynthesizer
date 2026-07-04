import { useEffect } from "react";
import { useAppStore } from "../../../store/app";
import {
  useVoiceModelStore,
  voiceVersionBadge,
  voiceSpeakerOptions,
  formatSampleRateKhz,
  type VoiceModelEntry,
  type VoiceType,
} from "../../../store/voice-models";
import { t18, type I18nText } from "../../../lib/models/msst-catalog";

/** Strings shared by BOTH voice nodes (RVC + SoVITS) — node-specific ones stay in the nodes. */
export const VOICE_STRINGS = {
  f0Shift: { zh: "变调", en: "Pitch", ja: "ピッチ" },
  f0ShiftTip: { zh: "音高平移（半音），+12 = 升一个八度", en: "Pitch shift in semitones, +12 = one octave up", ja: "ピッチシフト（半音）、+12 = 1オクターブ上" },
  noise: { zh: "噪声", en: "Noise", ja: "ノイズ" },
  noiseTip: { zh: "合成随机性（noise_scale）", en: "Synthesis randomness (noise_scale)", ja: "合成のランダム性（noise_scale）" },
  off: { zh: "关", en: "Off", ja: "オフ" },
} satisfies Record<string, I18nText>;

/**
 * Resolve a voice node's selected model from its GRAPH params against the installed list, and
 * keep the persisted `voiceName` / `modelPath` in sync. Derived from params, never mirrored
 * into local state — same rule as SeparationNode: the modal-local undo restores node params via
 * setNodes WITHOUT remounting, and a useState mirror would keep showing (and re-committing) the
 * undone selection.
 */
export function useVoiceModelSelection(
  voiceType: VoiceType,
  params: Record<string, unknown>,
  updateParams: (updates: Record<string, unknown>) => void,
): { models: VoiceModelEntry[]; selected: VoiceModelEntry | undefined } {
  const models = useVoiceModelStore((s) => s.models[voiceType]);
  useEffect(() => { void useVoiceModelStore.getState().fetchModels(); }, []);

  const selectedName = (params.voiceName as string) ?? models[0]?.name ?? "";
  const selected = models.find((m) => m.name === selectedName) ?? models[0];

  // Persist the RESOLVED selection whenever it drifts from the params: first mount (auto-pick
  // the first installed model), a deleted model falling back, or a moved models dir changing
  // `path`. A real model SWITCH also resets speaker_id — a stale index from the previous model
  // could exceed the new one's n_speakers.
  useEffect(() => {
    if (!selected) return;
    if (params.voiceName !== selected.name || params.modelPath !== selected.path) {
      updateParams({
        voiceName: selected.name,
        modelPath: selected.path,
        ...(params.voiceName !== undefined && params.voiceName !== selected.name
          ? { speaker_id: null }
          : {}),
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.name, selected?.path, params.voiceName, params.modelPath]);

  return { models, selected };
}

/**
 * Model dropdown + meta row (version badge / sample rate / index tag / speaker count), or the
 * "no models installed → go import" empty state. onSelect gets the full entry so the node can
 * write voiceName + modelPath (+ reset speaker_id) in one params update.
 */
export function VoiceModelPicker({ models, selected, lang, onSelect }: {
  models: VoiceModelEntry[];
  selected: VoiceModelEntry | undefined;
  lang: string;
  onSelect: (m: VoiceModelEntry) => void;
}) {
  const toggleModelManager = useAppStore((s) => s.toggleModelManager);

  if (models.length === 0) {
    return (
      <div className="voice-no-model">
        <span className="sep-no-model">
          {t18({ zh: "未安装模型", en: "No models installed", ja: "モデル未インストール" }, lang)}
        </span>
        <button className="voice-manage-btn" onClick={(e) => { e.stopPropagation(); toggleModelManager(); }}>
          {t18({ zh: "去资源管理导入", en: "Import in Resource Manager", ja: "リソース管理で取り込む" }, lang)}
        </button>
      </div>
    );
  }

  const badge = selected ? voiceVersionBadge(selected) : null;
  const speakerCount = selected ? voiceSpeakerOptions(selected).length : 0;

  return (
    <>
      <select
        className="sep-model-select"
        value={selected?.name ?? ""}
        onChange={(e) => {
          const m = models.find((x) => x.name === e.target.value);
          if (m) onSelect(m);
        }}
      >
        {models.map((m) => (
          <option key={m.name} value={m.name}>{m.name}</option>
        ))}
      </select>
      {selected && (
        <div className="voice-model-meta">
          {badge && <span className="ver-badge">{badge}</span>}
          <span>{formatSampleRateKhz(selected.sample_rate)}</span>
          {selected.index_path && (
            <span className="ver-badge" title={t18({ zh: "已附带检索/聚类文件", en: "Index/cluster asset present", ja: "インデックス/クラスタあり" }, lang)}>
              IDX
            </span>
          )}
          {speakerCount > 1 && (
            <span>{speakerCount} {t18({ zh: "说话人", en: "speakers", ja: "話者" }, lang)}</span>
          )}
        </div>
      )}
    </>
  );
}

/** Speaker dropdown row — renders NOTHING for single-speaker models (contract: null = 0). */
export function SpeakerSelect({ model, value, onChange, lang }: {
  model: VoiceModelEntry | undefined;
  value: number | null;
  onChange: (id: number) => void;
  lang: string;
}) {
  const opts = model ? voiceSpeakerOptions(model) : [];
  if (opts.length === 0) return null;
  return (
    <div className="sep-param-row">
      <label title={t18({ zh: "多说话人模型的目标说话人", en: "Target speaker of a multi-speaker model", ja: "マルチスピーカーモデルの話者" }, lang)}>
        {t18({ zh: "说话人", en: "Speaker", ja: "話者" }, lang)}
      </label>
      <select value={String(value ?? 0)} onChange={(e) => onChange(parseInt(e.target.value, 10))}>
        {opts.map((o) => (
          <option key={o.id} value={o.id}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}
