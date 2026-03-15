import React, { useState, useMemo } from "react";
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
  series_role: string;
}

interface MixedProbeData {
  data_rows: DataRow[];
}

const LLM_COLORS: Record<string, string> = {
  "Apertus-8B-Instruct-2509": "hsl(16, 70%, 52%)",   // orange
  "Apertus-8B-2509":          "hsl(280, 50%, 50%)",  // purple
  "Llama-3.1-8B-Instruct":    "hsl(220, 60%, 45%)",  // blue
  "Llama-3.1-8B":             "hsl(150, 50%, 38%)",  // green
};

const LLM_SHORT: Record<string, string> = {
  "Apertus-8B-Instruct-2509": "Apertus-Instruct",
  "Apertus-8B-2509":          "Apertus-8B-2509",
  "Llama-3.1-8B-Instruct":    "Llama-Instruct",
  "Llama-3.1-8B":             "Llama-3.1-8B",
};

const LineTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const items = payload.filter((p: any) => p.value != null && p.dataKey !== "__dummy");
  if (!items.length) return null;
  return (
    <div style={{
      background: "rgba(255,255,255,0.97)", border: "1px solid hsl(220,10%,85%)",
      borderRadius: 6, padding: "6px 10px", fontFamily: "Inter,sans-serif",
      fontSize: 11, boxShadow: "0 2px 8px rgba(0,0,0,.10)", maxWidth: 240, pointerEvents: "none",
    }}>
      <div style={{ fontWeight: 600, color: "#444", marginBottom: 4 }}>Layer {label}</div>
      {items.map((p: any) => (
        <div key={p.dataKey} style={{ display: "flex", justifyContent: "space-between", gap: 12, lineHeight: 1.6 }}>
          <span style={{ color: p.color, whiteSpace: "nowrap" }}>{p.name}</span>
          <span style={{ color: "#555", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
};

export default function MixedProbeLineChart({ data }: { data: MixedProbeData }) {
  const allRows = useMemo(() => data.data_rows, [data]);

  const llms = useMemo(
    () => [...new Set<string>(allRows.filter(r => r.llm_name).map(r => r.llm_name))],
    [allRows]
  );

  const layers = useMemo(
    () => [...new Set<number>(allRows.map(r => r.layer))].sort((a, b) => a - b),
    [allRows]
  );

  const [selectedModels, setSelectedModels] = useState<string[]>(llms);

  const toggleModel = (model: string) => {
    setSelectedModels(prev => 
      prev.includes(model) 
        ? prev.filter(m => m !== model)
        : [...prev, model].sort((a, b) => llms.indexOf(a) - llms.indexOf(b))
    );
  };

  const chartData = useMemo(() => {
    return layers.map(layer => {
      const pt: Record<string, number | string> = { layer };
      selectedModels.forEach(llm => {
        const l001 = allRows.find(r => r.llm_name === llm && r.probe_model === "L-0.01" && r.layer === layer && r.metric === "RMSE");
        if (l001) pt[`${llm}_L-0.01`] = l001.value;
        const logit = allRows.find(r => r.llm_name === llm && r.probe_model === "Logit-L-0.01" && r.layer === layer && r.metric === "RMSE");
        if (logit) pt[`${llm}_Logit`] = logit.value;
      });
      const dummy = allRows.find(r => r.layer === layer && r.series_role === "reference_series");
      if (dummy) pt["__dummy"] = dummy.value;
      return pt;
    });
  }, [allRows, layers, selectedModels]);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-2 border-b pb-4">
        <span className="font-sans text-xs font-semibold text-muted-foreground uppercase tracking-wide mr-1 hidden sm:inline">Models:</span>
        {llms.map(llm => {
          const isSelected = selectedModels.includes(llm);
          return (
            <button
              key={llm}
              onClick={() => toggleModel(llm)}
              className={`px-3 py-1.5 flex items-center gap-2 rounded-full font-sans text-xs border transition-all ${
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

      {/* Main Chart */}
      <div className="rounded-xl border bg-card p-4 sm:p-6 shadow-sm">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(40,12%,91%)" />
            <XAxis dataKey="layer" tick={{ fontFamily: "Inter", fontSize: 12 }}
              label={{ value: "Layer", position: "insideBottom", offset: -10, style: { fontFamily: "Inter", fontSize: 12 } }} />
            <YAxis tick={{ fontFamily: "Inter", fontSize: 12 }} domain={["auto", "auto"]} width={50}
              label={{ value: "RMSE", angle: -90, position: "insideLeft", style: { fontFamily: "Inter", fontSize: 13 } }} />
            <Tooltip content={<LineTooltip />}
              cursor={{ stroke: "hsl(220,10%,70%)", strokeWidth: 1, strokeDasharray: "3 2" }} />
            
            {/* Render lines for selected models */}
            {selectedModels.map(llm => (
              <React.Fragment key={llm}>
                <Line type="monotone" dataKey={`${llm}_L-0.01`} name={`${LLM_SHORT[llm]} L-0.01`}
                  stroke={LLM_COLORS[llm] ?? "hsl(220,30%,50%)"}
                  strokeWidth={2} dot={false} activeDot={{ r: 5 }} connectNulls />
                <Line type="monotone" dataKey={`${llm}_Logit`} name={`${LLM_SHORT[llm]} Logit`}
                  stroke={LLM_COLORS[llm] ?? "hsl(220,30%,50%)"}
                  strokeWidth={2} strokeDasharray="5 5" dot={false} activeDot={{ r: 4 }} connectNulls />
              </React.Fragment>
            ))}

            {/* Dummy baseline */}
            <Line type="monotone" dataKey="__dummy" name="Dummy baseline"
              stroke="hsl(220,10%,65%)" strokeWidth={1.5} strokeDasharray="6 3"
              dot={false} activeDot={false} legendType="none" />
          </LineChart>
        </ResponsiveContainer>

        {/* Legend describing line styles */}
        <div className="flex flex-wrap justify-center gap-x-6 gap-y-2 mt-4 font-sans text-[11px] text-muted-foreground border-t pt-4">
          <span className="flex items-center gap-2">
            <svg width="24" height="10"><line x1="0" y1="5" x2="24" y2="5" stroke="currentColor" strokeWidth="2.5"/></svg>
            L-0.01
          </span>
          <span className="flex items-center gap-2">
            <svg width="24" height="10"><line x1="0" y1="5" x2="24" y2="5" stroke="currentColor" strokeWidth="2" strokeDasharray="4 3"/></svg>
            Logit-L-0.01
          </span>
          <span className="flex items-center gap-2">
            <svg width="24" height="10"><line x1="0" y1="5" x2="24" y2="5" stroke="currentColor" strokeWidth="1.5" strokeDasharray="6 3"/></svg>
            Dummy RMSE
          </span>
        </div>
      </div>
    </div>
  );
}
