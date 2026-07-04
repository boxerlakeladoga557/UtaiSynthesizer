import { useCallback } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider, formatRatio } from "./ParamSlider";
import { VoiceModelPicker, SpeakerSelect, useVoiceModelSelection, VOICE_STRINGS } from "./VoiceModelPicker";
import { SOVITS_DEFAULTS } from "../../../lib/workflow/voiceDefaults";
import type { VoiceModelEntry } from "../../../store/voice-models";
import { t18 } from "../../../lib/models/msst-catalog";

// NO shallow-diffusion checkbox: deferred by user decision (with auto-f0 / spk-mix) — no dead UI.
export function SoVitsNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [params, updateParams] = useNodeParams(props);
  const { models, selected } = useVoiceModelSelection("sovits", params, updateParams);

  // Param keys ARE the wire contract keys (see voiceDefaults.ts) — absent = contract default.
  const f0Shift = (params.f0_shift as number) ?? SOVITS_DEFAULTS.f0_shift;
  const noiseScale = (params.noise_scale as number) ?? SOVITS_DEFAULTS.noise_scale;
  const clusterRatio = (params.cluster_ratio as number) ?? SOVITS_DEFAULTS.cluster_ratio;
  const loudnessEnvelope = (params.loudness_envelope as number) ?? SOVITS_DEFAULTS.loudness_envelope;
  const speakerId = (params.speaker_id as number | null) ?? SOVITS_DEFAULTS.speaker_id;
  // Cluster/index asset presence comes from the SAME ModelEntry field RVC's index uses (Rust
  // scan() picks up any sibling .npy regardless of model type).
  const hasCluster = !!selected?.index_path;

  const handleSelect = useCallback((m: VoiceModelEntry) => {
    // Model switch resets the speaker — a stale index could exceed the new model's n_speakers.
    updateParams({ voiceName: m.name, modelPath: m.path, speaker_id: null });
  }, [updateParams]);

  return (
    <NodeShell nodeId={props.id} label="SoVITS" icon="[S]" color="#8b5cf6" inputs={1} outputs={1}>
      <div className="sep-node-body">
        <VoiceModelPicker models={models} selected={selected} lang={lang} onSelect={handleSelect} />
        {models.length > 0 && (
          <div className="sep-params">
            <ParamSlider
              label={t18(VOICE_STRINGS.f0Shift, lang)}
              title={t18(VOICE_STRINGS.f0ShiftTip, lang)}
              min={-24} max={24} step={1} value={f0Shift}
              onChange={(v) => updateParams({ f0_shift: v })}
            />
            <ParamSlider
              label={t18(VOICE_STRINGS.noise, lang)}
              title={t18(VOICE_STRINGS.noiseTip, lang)}
              min={0} max={1} step={0.01} value={noiseScale} format={formatRatio}
              onChange={(v) => updateParams({ noise_scale: v })}
            />
            <ParamSlider
              label={t18({ zh: "聚类占比", en: "Cluster ratio", ja: "クラスタ率" }, lang)}
              title={hasCluster
                ? t18({ zh: "聚类/特征检索混合比例：越高越像目标音色，咬字可能变糊；0 = 关闭", en: "Cluster / feature-index blend — higher = closer to the target timbre, may slur articulation; 0 = off", ja: "クラスタ/特徴検索の混合比 — 高いほど目標声質に近づくが発音が不明瞭になることも。0 = 無効" }, lang)
                : t18({ zh: "该模型没有聚类/检索文件（导入时附带可启用）", en: "This model has no cluster/index asset (import one alongside to enable)", ja: "このモデルにはクラスタ/インデックスがありません（併せて取り込むと有効化）" }, lang)}
              disabled={!hasCluster}
              min={0} max={1} step={0.01} value={clusterRatio} format={formatRatio}
              onChange={(v) => updateParams({ cluster_ratio: v })}
            />
            <ParamSlider
              label={t18({ zh: "响度包络", en: "Loudness env", ja: "音量包絡" }, lang)}
              title={t18({ zh: "用输入响度包络替换输出的混合比例，1 = 不替换（关）", en: "Input-loudness-envelope replacement mix; 1 = no replacement (off)", ja: "入力のラウドネス包絡で出力を置き換える比率。1 = 置き換えなし（オフ）" }, lang)}
              min={0} max={1} step={0.01} value={loudnessEnvelope}
              format={(v) => (v >= 1 ? t18(VOICE_STRINGS.off, lang) : v.toFixed(2))}
              onChange={(v) => updateParams({ loudness_envelope: v })}
            />
            <SpeakerSelect model={selected} value={speakerId} lang={lang}
              onChange={(id) => updateParams({ speaker_id: id })} />
          </div>
        )}
      </div>
    </NodeShell>
  );
}
