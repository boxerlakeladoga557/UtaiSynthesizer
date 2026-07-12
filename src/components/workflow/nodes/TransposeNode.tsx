import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
import { t18 } from "../../../lib/models/msst-catalog";

/** Fidelity transpose (spectral pitch-shift, Signalsmith Stretch — formant/tonality-aware,
 *  polyphonic-safe). Built for INSTRUMENTAL transposition: the voice nodes already transpose
 *  model-side (f0_shift), but nothing could shift the accompaniment until this node.
 *  0 semitones = exact passthrough (the engine skips the invoke entirely). */
export function TransposeNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [params, updateParams] = useNodeParams(props);

  const semitones = (params.semitones as number) ?? 0;

  return (
    <NodeShell nodeId={props.id} label={t18({ zh: "移调", en: "Transpose", ja: "移調" }, lang)} icon="[T]" color="#fbbf24" inputs={1} outputs={1}>
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t18({ zh: "半音", en: "Semitones", ja: "半音" }, lang)}
            title={t18({
              zh: "频谱域保真移调（保持时长与共振峰特性，适合给伴奏整体移调）；0 = 原样直通",
              en: "Spectral-domain fidelity pitch shift (length & tonality preserved — made for transposing accompaniment); 0 = exact passthrough",
              ja: "スペクトル領域の高品質ピッチシフト（長さと音色を保持、伴奏の移調向け）。0 = そのまま通過",
            }, lang)}
            min={-24} max={24} step={1} value={semitones}
            format={(v) => (v > 0 ? `+${v}` : `${v}`)}
            onChange={(v) => updateParams({ semitones: v })}
          />
        </div>
      </div>
    </NodeShell>
  );
}
