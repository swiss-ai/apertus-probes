import { useState, useMemo, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// Uses robustness.json — all 4 LLMs, L-0.01 probe, exact+last token positions, 6 datasets
// Structure matches PDF Figure 1: rows=token_position, columns=dataset, lines=LLMs
interface DataRow {
  llm_name: string;
  probe_model: string;
  token_position: string;  // "exact" | "last"
  layer: number;
  value: number;
  panel_row: number;  // 0=exact, 1=last
  panel_col: number;  // 0-5 dataset index
  panel_title: string;
  panel_id: string;
  series_role: string;
}

interface RobustnessData {
  data_rows: DataRow[];
  figure?: { title?: string };
}

// Columns ordered as in PDF
const DATASET_COLS = [
  "MMLU High School",
  "MMLU Professional",
  "ARC Challenge",
  "ARC Easy",
  "SMS Spam",
  "Finance YesNo",
];

const DATASET_SHORT: Record<string, string> = {
  "MMLU High School":  "MMLU HS",
  "MMLU Professional": "MMLU Prof",
  "ARC Challenge":     "ARC Chall",
  "ARC Easy":          "ARC Easy",
  "SMS Spam":          "SMS Spam",
  "Finance YesNo":     "Finance YesNo",
};

const TOKEN_LABELS: Record<number, string> = {
  0: "Exact token",
  1: "Last token",
};

// LLM colors matching the PDF style
const LLM_COLORS: Record<string, string> = {
  "Apertus-8B-Instruct-2509": "hsl(16, 70%, 52%)",   // orange
  "Apertus-8B-2509":          "hsl(280, 50%, 50%)",  // purple
  "Llama-3.1-8B-Instruct":    "hsl(220, 60%, 45%)",  // blue
  "Llama-3.1-8B":             "hsl(150, 50%, 38%)",  // green
};

const LLM_SHORT: Record<string, string> = {
  "Apertus-8B-Instruct-2509": "Apertus-Instruct L-0.01",
  "Apertus-8B-2509":          "Apertus-8B-2509 L-0.01",
  "Llama-3.1-8B-Instruct":    "Llama-Instruct L-0.01",
  "Llama-3.1-8B":             "Llama-3.1-8B L-0.01",
};

const LineTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const items = payload.filter((p: any) => p.value != null && p.name !== "__dummy");
  if (!items.length) return null;
  return (
    <div style={{
      background: "rgba(255,255,255,0.97)", border: "1px solid hsl(220,10%,85%)",
      borderRadius: 6, padding: "4px 8px", fontFamily: "Inter,sans-serif",
      fontSize: 10, boxShadow: "0 1px 6px rgba(0,0,0,.10)", maxWidth: 190, pointerEvents: "none",
    }}>
      <div style={{ fontWeight: 600, color: "#444", marginBottom: 2 }}>Layer {label}</div>
      {items.map((p: any) => (
        <div key={p.dataKey} style={{ display: "flex", justifyContent: "space-between", gap: 8, lineHeight: 1.5 }}>
          <span style={{ color: p.color, whiteSpace: "nowrap" }}>{LLM_SHORT[p.name] ?? p.name}</span>
          <span style={{ color: "#555", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
};

function CellChart({
  chartData,
  llms,
  height,
  fontSize = 8,
}: {
  chartData: Record<string, number | string>[];
  llms: string[];
  height: number;
  fontSize?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={{ top: 3, right: 4, bottom: 18, left: -14 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(40,12%,91%)" />
        <XAxis dataKey="layer" tick={{ fontFamily: "Inter", fontSize }}
          label={{ value: "Layer", position: "insideBottom", offset: -10, style: { fontFamily: "Inter", fontSize } }} />
        <YAxis tick={{ fontFamily: "Inter", fontSize }} domain={["auto", "auto"]} width={32} />
        <Tooltip content={<LineTooltip />}
          cursor={{ stroke: "hsl(220,10%,70%)", strokeWidth: 1, strokeDasharray: "3 2" }} />
        {llms.map(llm => (
          <Line key={llm} type="monotone" dataKey={llm} name={llm}
            stroke={LLM_COLORS[llm] ?? "hsl(220,30%,50%)"}
            strokeWidth={1.5} dot={false} activeDot={{ r: 2.5 }} connectNulls />
        ))}
        <Line type="monotone" dataKey="__dummy" name="__dummy"
          stroke="hsl(220,10%,65%)" strokeWidth={1} strokeDasharray="5 3"
          dot={false} activeDot={false} legendType="none" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function LayerRMSEChart({ data }: { data: RobustnessData }) {
  const allRows = useMemo(() => data.data_rows, [data]);

  const llms = useMemo(
    () => [...new Set<string>(allRows.filter(r => r.series_role === "primary_series").map(r => r.llm_name))],
    [allRows]
  );

  const tokenRows = useMemo(() => {
    const rowNums = [...new Set<number>(allRows.map(r => r.panel_row ?? 0))].sort();
    return rowNums;
  }, [allRows]);

  const layers = useMemo(
    () => [...new Set<number>(allRows.map(r => r.layer))].sort((a, b) => a - b),
    [allRows]
  );

  const [selectedTokenPos, setSelectedTokenPos] = useState<number>(0);
  const [selectedModels, setSelectedModels] = useState<string[]>(llms);
  const [expanded, setExpanded] = useState<{ panelRow: number; panelCol: number } | null>(null);

  const toggleModel = (model: string) => {
    setSelectedModels(prev => 
      prev.includes(model) 
        ? prev.filter(m => m !== model)
        : [...prev, model].sort((a, b) => llms.indexOf(a) - llms.indexOf(b))
    );
  };

  const handleKey = useCallback((e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(null); }, []);
  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  const buildChartData = useCallback((panelRow: number, panelCol: number) => {
    const primary = allRows.filter(r => (r.panel_row ?? 0) === panelRow && (r.panel_col ?? 0) === panelCol && r.series_role === "primary_series");
    const ref     = allRows.filter(r => (r.panel_row ?? 0) === panelRow && (r.panel_col ?? 0) === panelCol && r.series_role === "reference_series");
    return layers.map(layer => {
      const pt: Record<string, number | string> = { layer };
      selectedModels.forEach(llm => {
        const m = primary.find(r => r.llm_name === llm && r.layer === layer);
        if (m) pt[llm] = m.value;
      });
      const d = ref.find(r => r.layer === layer);
      if (d) pt["__dummy"] = d.value;
      return pt;
    });
  }, [allRows, layers, selectedModels]);

  // Get panel title by (row, col)
  const getPanelTitle = (panelRow: number, panelCol: number) => {
    const row = allRows.find(r => (r.panel_row ?? 0) === panelRow && (r.panel_col ?? 0) === panelCol);
    return row?.panel_title ?? row?.panel_id ?? `Panel (${panelRow},${panelCol})`;
  };

  const expandedData = expanded ? buildChartData(expanded.panelRow, expanded.panelCol) : null;
  const expandedTitle = expanded ? getPanelTitle(expanded.panelRow, expanded.panelCol) : "";

  // Get the dataset columns that actually exist
  const panelCols = useMemo(
    () => [...new Set<number>(allRows.map(r => r.panel_col ?? 0))].sort(),
    [allRows]
  );

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between border-b pb-4">
        {/* Token Position Toggle */}
        <div className="flex items-center gap-3 shrink-0">
          <span className="font-sans text-xs font-semibold text-muted-foreground uppercase tracking-wide">Position:</span>
          <div className="flex space-x-1 border rounded-lg p-1 bg-muted/30">
            {tokenRows.map(rowNum => (
              <button
                key={rowNum}
                onClick={() => setSelectedTokenPos(rowNum)}
                className={`px-3 py-1 text-xs font-sans rounded-md transition-colors ${
                  selectedTokenPos === rowNum 
                    ? "bg-background shadow-sm border text-foreground" 
                    : "text-muted-foreground hover:bg-muted/50 border border-transparent"
                }`}
              >
                {TOKEN_LABELS[rowNum] ?? `Row ${rowNum}`}
              </button>
            ))}
          </div>
        </div>

        {/* Model Multi-Selector */}
        <div className="flex flex-wrap items-center justify-start sm:justify-end gap-2 w-full">
          <span className="font-sans text-xs font-semibold text-muted-foreground uppercase tracking-wide mr-1 hidden sm:inline">Models:</span>
          {llms.map(llm => {
            const isSelected = selectedModels.includes(llm);
            return (
              <button
                key={llm}
                onClick={() => toggleModel(llm)}
                className={`px-2.5 py-1 flex items-center gap-2 rounded-full font-sans text-xs border transition-all ${
                  isSelected 
                    ? "bg-background shadow-sm text-foreground" 
                    : "bg-muted/30 text-muted-foreground/60 border-transparent hover:bg-muted/50"
                }`}
                style={{ borderColor: isSelected ? LLM_COLORS[llm] : undefined }}
              >
                <div 
                  className="w-2.5 h-2.5 rounded-full transition-opacity shrink-0" 
                  style={{ backgroundColor: LLM_COLORS[llm] ?? "#888", opacity: isSelected ? 1 : 0.3 }}
                />
                <span className="truncate max-w-[140px] sm:max-w-none">{LLM_SHORT[llm] ?? llm}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Grid for selected Token Position */}
      <div className="space-y-1">
          <div className="font-sans text-xs font-semibold text-muted-foreground uppercase tracking-wide border-b pb-1">
            RMSE ({TOKEN_LABELS[selectedTokenPos] ?? `Row ${selectedTokenPos}`})
          </div>

          {/* Dataset column grid */}
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-y-6 gap-x-4">
            {panelCols.map(colNum => {
              const chartData = buildChartData(selectedTokenPos, colNum);
              const title = getPanelTitle(selectedTokenPos, colNum);
              const shortTitle = DATASET_SHORT[title] ?? title;
              return (
                <div key={colNum} className="flex flex-col group">
                  <div className="text-center font-sans text-[11px] font-semibold text-muted-foreground pb-1.5 truncate px-1" title={title}>
                    {shortTitle}
                  </div>
                  <div
                    className="rounded border bg-card cursor-zoom-in group-hover:border-foreground/30 transition-colors flex-1"
                    onClick={() => setExpanded({ panelRow: selectedTokenPos, panelCol: colNum })}
                    title={`Click to expand — ${title}`}
                  >
                    <CellChart chartData={chartData} llms={selectedModels} height={200} />
                  </div>
                </div>
              );
            })}
          </div>
      </div>

      {/* Shared legend (Simplified since models act as legend) */}
      <div className="flex flex-wrap justify-center gap-x-5 gap-y-1 font-sans text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <svg width="18" height="7"><line x1="0" y1="3.5" x2="18" y2="3.5" stroke="hsl(220,10%,65%)" strokeWidth="1.5" strokeDasharray="5 3"/></svg>
          Dummy RMSE
        </span>
        <span className="text-muted-foreground/50 italic ml-2">Click any panel to expand ↗</span>
      </div>

      {/* Expanded modal */}
      {expanded && expandedData && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          style={{ background: "rgba(0,0,0,0.45)", backdropFilter: "blur(3px)" }}
          onClick={() => setExpanded(null)}
        >
          <div
            className="bg-card rounded-2xl shadow-2xl border w-full max-w-3xl p-6 space-y-3"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-sans text-sm font-semibold">
                {expandedTitle}
                <span className="ml-2 font-normal text-xs text-muted-foreground">
                  — {TOKEN_LABELS[expanded.panelRow]} · L-0.01
                </span>
              </h3>
              <button onClick={() => setExpanded(null)}
                className="rounded-full w-7 h-7 flex items-center justify-center text-muted-foreground hover:bg-muted transition-colors font-bold">✕</button>
            </div>

            <ResponsiveContainer width="100%" height={380}>
              <LineChart data={expandedData} margin={{ top: 6, right: 16, bottom: 28, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(40,12%,91%)" />
                <XAxis dataKey="layer" tick={{ fontFamily: "Inter", fontSize: 11 }}
                  label={{ value: "Layer", position: "insideBottom", offset: -14, style: { fontFamily: "Inter", fontSize: 11 } }} />
                <YAxis tick={{ fontFamily: "Inter", fontSize: 11 }} domain={["auto", "auto"]} width={42}
                  label={{ value: "RMSE", angle: -90, position: "insideLeft", offset: 10, style: { fontFamily: "Inter", fontSize: 11 } }} />
                <Tooltip content={<LineTooltip />}
                  cursor={{ stroke: "hsl(220,10%,70%)", strokeWidth: 1, strokeDasharray: "3 2" }} />
                {selectedModels.map(llm => (
                  <Line key={llm} type="monotone" dataKey={llm} name={llm}
                    stroke={LLM_COLORS[llm] ?? "hsl(220,30%,50%)"}
                    strokeWidth={2} dot={false} activeDot={{ r: 4 }} connectNulls />
                ))}
                <Line type="monotone" dataKey="__dummy" name="__dummy"
                  stroke="hsl(220,10%,65%)" strokeWidth={1.5} strokeDasharray="6 3"
                  dot={false} activeDot={false} legendType="none" />
              </LineChart>
            </ResponsiveContainer>

            <div className="flex flex-wrap justify-center gap-x-5 gap-y-1 font-sans text-[10px] text-muted-foreground">
              {selectedModels.map(llm => (
                <span key={llm} className="flex items-center gap-1.5">
                  <svg width="18" height="7"><line x1="0" y1="3.5" x2="18" y2="3.5" stroke={LLM_COLORS[llm] ?? "#888"} strokeWidth="2"/></svg>
                  {LLM_SHORT[llm] ?? llm}
                </span>
              ))}
              <span className="flex items-center gap-1.5">
                <svg width="18" height="7"><line x1="0" y1="3.5" x2="18" y2="3.5" stroke="hsl(220,10%,65%)" strokeWidth="1.5" strokeDasharray="6 3"/></svg>
                Dummy RMSE
              </span>
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
