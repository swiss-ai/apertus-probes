import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";

interface DataRow {
  panel_id: string;
  panel_title: string;
  probe_type: string;
  view_type: string;
  category: string;
  category_order: number;
  method: string;
  value: number;
}

interface SPIBarData {
  figure: { title: string };
  legend: { legend_label: string; method: string }[];
  data_rows: DataRow[];
}

const BAR_COLORS: Record<string, string> = {
  "Baseline (prompting)": "hsl(220, 10%, 65%)",
  MERA: "hsl(220, 60%, 45%)",
  "MERA logistic": "hsl(190, 60%, 45%)",
  "MERA contrastive": "hsl(16, 70%, 52%)",
  "Baseline Error (Additive)": "hsl(45, 70%, 50%)",
};

// Custom tooltip: small pill showing only the one bar being hovered
const CustomBarTooltip = ({ active, payload, label }: any) => {
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
      maxWidth: 200,
      pointerEvents: "none",
    }}>
      <div style={{ fontWeight: 600, color: "#333", marginBottom: 3, fontSize: 10 }}>{label}</div>
      {items.map((p: any) => (
        <div key={p.name} style={{ display: "flex", justifyContent: "space-between", gap: 10, lineHeight: 1.6 }}>
          <span style={{ color: BAR_COLORS[p.name] || "#888", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 130 }}>
            {p.name}
          </span>
          <span style={{ color: "#444", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
            {Number(p.value).toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function SPIBarChart({ data }: { data: SPIBarData }) {
  const panels = useMemo(() => {
    const panelMap = new Map<string, { title: string; rows: DataRow[] }>();
    data.data_rows.forEach((r) => {
      if (!panelMap.has(r.panel_id)) {
        panelMap.set(r.panel_id, { title: r.panel_title, rows: [] });
      }
      panelMap.get(r.panel_id)!.rows.push(r);
    });
    return [...panelMap.entries()];
  }, [data]);

  const methods = data.legend.map((l) => l.method);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {panels.map(([panelId, { title, rows }]) => {
        const categories = [...new Set(rows.map((r) => r.category))];
        const chartData = categories.map((cat) => {
          const point: Record<string, string | number> = { category: cat };
          rows
            .filter((r) => r.category === cat)
            .forEach((r) => {
              point[r.method] = r.value;
            });
          return point;
        });

        return (
          <div key={panelId}>
            <h5 className="font-sans text-xs font-semibold text-muted-foreground mb-2 text-center">
              {title}
            </h5>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData} margin={{ top: 5, right: 10, bottom: 20, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(40, 12%, 88%)" />
                <XAxis
                  dataKey="category"
                  tick={{ fontFamily: "Inter", fontSize: 9 }}
                  angle={-30}
                  textAnchor="end"
                  height={50}
                />
                <YAxis
                  tick={{ fontFamily: "Inter", fontSize: 10 }}
                  label={{ value: "SPI", angle: -90, position: "insideLeft", style: { fontFamily: "Inter", fontSize: 10 } }}
                />
                <Tooltip
                  content={<CustomBarTooltip />}
                  cursor={{ fill: "hsl(220, 10%, 95%)" }}
                />
                <ReferenceLine y={0} stroke="hsl(220, 10%, 50%)" strokeWidth={1} />
                {methods.map((m) => (
                  <Bar
                    key={m}
                    dataKey={m}
                    fill={BAR_COLORS[m] || "hsl(220, 30%, 60%)"}
                    radius={[2, 2, 0, 0]}
                    maxBarSize={20}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      })}
      {/* Shared legend */}
      <div className="md:col-span-2 flex flex-wrap justify-center gap-4 mt-2">
        {methods.map((m) => (
          <div key={m} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ background: BAR_COLORS[m] || "hsl(220, 30%, 60%)" }}
            />
            <span className="font-sans text-[10px] text-muted-foreground">{m}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
