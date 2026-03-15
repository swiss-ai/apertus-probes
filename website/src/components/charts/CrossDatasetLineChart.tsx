import { useMemo, useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface DataRow {
  llm_name: string;
  probe_model: string;
  token_position: string;
  layer: number;
  metric: string;
  value: number;
  error_value?: number | null;
  panel_row: number;
  panel_col: number;
  train_dataset: string;
  test_dataset: string;
  train_dataset_title: string;
  test_dataset_title: string;
  panel_title: string;
  series_role: string;
  line_style: string;
  legend_label: string;
  color_group: string;
}

interface TransferData {
  figure: {
    title: string;
    panel_layout: { row_labels: string[]; col_labels: string[] };
  };
  data_rows: DataRow[];
}

const DATASET_ORDER = [
  "sms_spam",
  "sujet_finance_yesno_5k",
  "mmlu_high_school",
  "mmlu_professional",
  "ARC-Easy",
  "ARC-Challenge",
];

const DATASET_LABELS: Record<string, string> = {
  sms_spam: "SMS Spam",
  sujet_finance_yesno_5k: "Yes-No Finance",
  mmlu_high_school: "MMLU HS",
  mmlu_professional: "MMLU Prof",
  "ARC-Easy": "ARC-Easy",
  "ARC-Challenge": "ARC-Chall",
};

const ROW_COLORS: string[] = [
  "hsl(220, 60%, 45%)",
  "hsl(16, 70%, 52%)",
  "hsl(150, 50%, 38%)",
  "hsl(280, 50%, 50%)",
  "hsl(45, 80%, 45%)",
  "hsl(190, 60%, 40%)",
];

const LineTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || payload.length === 0) return null;
  const items = payload.filter((p: any) => p.value != null);
  if (items.length === 0) return null;
  return (
    <div style={{
      background: "rgba(255,255,255,0.96)",
      border: "1px solid hsl(220,10%,85%)",
      borderRadius: 6,
      padding: "4px 8px",
      fontFamily: "Inter, sans-serif",
      fontSize: 10,
      boxShadow: "0 1px 6px rgba(0,0,0,0.10)",
      maxWidth: 170,
      pointerEvents: "none",
    }}>
      <div style={{ fontWeight: 600, color: "#444", marginBottom: 2, fontSize: 10 }}>Layer {label}</div>
      {items.map((p: any) => (
        <div key={p.dataKey} style={{ display: "flex", justifyContent: "space-between", gap: 8, lineHeight: 1.5 }}>
          <span style={{ color: p.color, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 100 }}>
            {DATASET_LABELS[p.name] ?? p.name}
          </span>
          <span style={{ color: "#555", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
            {Number(p.value).toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
};

// Reusable chart for a single panel
function PanelChart({
  chartData,
  trainDs,
  testDs,
  height,
  margin,
  fontSize = 9,
}: {
  chartData: Record<string, number | string>[];
  trainDs: string[];
  testDs: string;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  fontSize?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={margin}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(40, 12%, 88%)" />
        <XAxis
          dataKey="layer"
          tick={{ fontFamily: "Inter", fontSize }}
          label={{ value: "Layer", position: "insideBottom", offset: -12, style: { fontFamily: "Inter", fontSize } }}
        />
        <YAxis
          tick={{ fontFamily: "Inter", fontSize }}
          domain={["auto", "auto"]}
          width={38}
        />
        <Tooltip
          content={<LineTooltip />}
          cursor={{ stroke: "hsl(220,10%,70%)", strokeWidth: 1, strokeDasharray: "3 2" }}
        />
        {DATASET_ORDER.filter((ds) => trainDs.includes(ds)).map((ds, i) => (
          <Line
            key={ds}
            type="monotone"
            dataKey={ds}
            name={ds}
            stroke={ROW_COLORS[i % ROW_COLORS.length]}
            strokeWidth={ds === testDs ? 2.5 : 1.5}
            strokeDasharray={ds === testDs ? undefined : "4 3"}
            dot={false}
            activeDot={{ r: 3 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function CrossDatasetLineChart({ data }: { data: TransferData }) {
  const allLLMs = useMemo(
    () => [...new Set<string>(data.data_rows.map((r) => r.llm_name))].sort(),
    [data]
  );

  const allProbeModels = useMemo(
    () => [...new Set<string>(data.data_rows.map((r) => r.probe_model))].sort(),
    [data]
  );

  const [selectedLLM, setSelectedLLM] = useState(() => allLLMs[0] ?? "");
  const [selectedProbe, setSelectedProbe] = useState(() => allProbeModels[0] ?? "");
  const [expandedDs, setExpandedDs] = useState<string | null>(null);

  // Close modal on Escape
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === "Escape") setExpandedDs(null);
  }, []);
  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const filteredRows = useMemo(
    () =>
      data.data_rows.filter(
        (r) =>
          r.series_role === "primary_series" &&
          r.llm_name === selectedLLM &&
          r.probe_model === selectedProbe
      ),
    [data, selectedLLM, selectedProbe]
  );

  const panels = useMemo(() => {
    return DATASET_ORDER.map((testDs) => {
      const panelRows = filteredRows.filter((r) => r.test_dataset === testDs);
      const layers = [...new Set<number>(panelRows.map((r) => r.layer))].sort((a, b) => a - b);
      const chartData = layers.map((layer) => {
        const point: Record<string, number | string> = { layer };
        DATASET_ORDER.forEach((trainDs) => {
          const match = panelRows.find((r) => r.layer === layer && r.train_dataset === trainDs);
          if (match) point[trainDs] = match.value;
        });
        return point;
      });
      const trainDs = [...new Set<string>(panelRows.map((r) => r.train_dataset))];
      return { testDs, chartData, trainDs };
    });
  }, [filteredRows]);

  const expandedPanel = expandedDs ? panels.find((p) => p.testDs === expandedDs) : null;

  return (
    <div className="space-y-4">
      {/* Selectors */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="font-sans text-xs font-semibold text-muted-foreground uppercase tracking-wide">Model:</span>
          {allLLMs.map((llm) => (
            <button
              key={llm}
              onClick={() => setSelectedLLM(llm)}
              className={`px-3 py-1 rounded-full font-sans text-xs border transition-colors ${
                selectedLLM === llm
                  ? "bg-foreground text-background border-foreground"
                  : "border-border text-muted-foreground hover:border-foreground/50"
              }`}
            >
              {llm}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <span className="font-sans text-xs font-semibold text-muted-foreground uppercase tracking-wide">Probe:</span>
          {allProbeModels.map((pm) => (
            <button
              key={pm}
              onClick={() => setSelectedProbe(pm)}
              className={`px-3 py-1 rounded-full font-sans text-xs border transition-colors ${
                selectedProbe === pm
                  ? "bg-foreground text-background border-foreground"
                  : "border-border text-muted-foreground hover:border-foreground/50"
              }`}
            >
              {pm}
            </button>
          ))}
        </div>
      </div>

      {/* Click-to-expand hint */}
      <p className="font-sans text-[10px] text-muted-foreground/60 text-right">
        Click any panel to expand ↗
      </p>

      {/* Grid of panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {panels.map(({ testDs, chartData, trainDs }) => (
          <div
            key={testDs}
            className="cursor-zoom-in rounded-lg border bg-card p-2 hover:border-foreground/30 transition-colors group"
            onClick={() => setExpandedDs(testDs)}
            title="Click to expand"
          >
            <h5 className="font-sans text-xs font-semibold text-muted-foreground mb-1 text-center flex items-center justify-center gap-1">
              Test: {DATASET_LABELS[testDs] ?? testDs}
              <span className="opacity-0 group-hover:opacity-50 transition-opacity text-[9px]">⤢</span>
            </h5>
            <PanelChart
              chartData={chartData}
              trainDs={trainDs}
              testDs={testDs}
              height={190}
              margin={{ top: 4, right: 8, bottom: 20, left: -5 }}
            />
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-x-5 gap-y-1 mt-2">
        {DATASET_ORDER.map((ds, i) => (
          <div key={ds} className="flex items-center gap-1.5">
            <svg width="18" height="7"><line x1="0" y1="3.5" x2="18" y2="3.5" stroke={ROW_COLORS[i] ?? "#888"} strokeWidth="1.8" /></svg>
            <span className="font-sans text-[10px] text-muted-foreground mr-1">
              Train: {DATASET_LABELS[ds] ?? ds}
            </span>
          </div>
        ))}
        <div className="flex items-center gap-1.5">
          <svg width="18" height="7"><line x1="0" y1="3.5" x2="18" y2="3.5" stroke="hsl(220,10%,65%)" strokeWidth="2.5" /></svg>
          <span className="font-sans text-[10px] text-muted-foreground italic">Bold = same-dataset (diagonal)</span>
        </div>
      </div>

      {/* Expanded modal */}
      {expandedPanel && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          style={{ background: "rgba(0,0,0,0.45)", backdropFilter: "blur(3px)" }}
          onClick={() => setExpandedDs(null)}
        >
          <div
            className="bg-card rounded-2xl shadow-2xl border w-full max-w-3xl p-6 space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between">
              <h3 className="font-sans text-sm font-semibold">
                Test: {DATASET_LABELS[expandedPanel.testDs] ?? expandedPanel.testDs}
                <span className="ml-2 text-muted-foreground font-normal text-xs">
                  — {selectedLLM} / {selectedProbe}
                </span>
              </h3>
              <button
                onClick={() => setExpandedDs(null)}
                className="rounded-full w-7 h-7 flex items-center justify-center text-muted-foreground hover:bg-muted transition-colors text-sm font-bold"
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            {/* Large chart */}
            <PanelChart
              chartData={expandedPanel.chartData}
              trainDs={expandedPanel.trainDs}
              testDs={expandedPanel.testDs}
              height={380}
              margin={{ top: 6, right: 16, bottom: 28, left: 0 }}
              fontSize={11}
            />

            {/* Modal legend */}
            <div className="flex flex-wrap justify-center gap-x-5 gap-y-1 pt-1">
              {DATASET_ORDER.filter((ds) => expandedPanel.trainDs.includes(ds)).map((ds, i) => (
                <div key={ds} className="flex items-center gap-1.5">
                  <svg width="18" height="7">
                    <line 
                      x1="0" y1="3.5" x2="18" y2="3.5" 
                      stroke={ROW_COLORS[i] ?? "#888"} 
                      strokeWidth={ds === expandedPanel.testDs ? "2.5" : "1.5"} 
                      strokeDasharray={ds === expandedPanel.testDs ? "none" : "4 3"} 
                    />
                  </svg>
                  <span className="font-sans text-[10px] text-muted-foreground mr-1">
                    Train: {DATASET_LABELS[ds] ?? ds}
                    {ds === expandedPanel.testDs && " ★"}
                  </span>
                </div>
              ))}
            </div>

            <p className="text-center text-[10px] text-muted-foreground/50">
              Press <kbd className="px-1 py-0.5 rounded border text-[9px]">Esc</kbd> or click outside to close
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
