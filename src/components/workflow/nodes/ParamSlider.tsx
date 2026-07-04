import "./NodeShell.css";

/**
 * Shared thin-bar slider row for workflow-node params (label + range + value readout) — ONE
 * source of truth for the markup that was previously pasted per-param inside SeparationNode.
 * The `sep-*` classes in NodeShell.css carry the S24 slider-specificity fix (`.sep-overlap`
 * prefix beats the global `input[type="range"]` thumb): reuse them, never copy the CSS.
 */
export function ParamSlider({ label, title, min, max, step, value, onChange, format, disabled }: {
  label: string;
  /** Tooltip on the label (and, when `disabled`, the reason the control is off). */
  title?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  /** Custom readout (e.g. protect >= 0.5 → "关"); default String(value). */
  format?: (value: number) => string;
  disabled?: boolean;
}) {
  return (
    <div className="sep-param-row">
      <label title={title}>{label}</label>
      <span className="sep-overlap nodrag">
        <input
          className="sep-overlap-range nodrag"
          type="range" min={min} max={max} step={step} value={value}
          disabled={disabled}
          onPointerDown={(e) => e.stopPropagation()}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
        <span className="sep-overlap-val">{format ? format(value) : String(value)}</span>
      </span>
    </div>
  );
}

/** Two-decimal readout for 0..1 ratio sliders. */
export function formatRatio(v: number): string {
  return v.toFixed(2);
}
