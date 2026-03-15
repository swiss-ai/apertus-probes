import { useMemo } from "react";

interface DataRow {
  Model: string;
  Dataset: string;
  "Probe Type": string;
  Method: string;
  value: number;
  panel_row: number;
  panel_col: number;
  panel_title: string;
  row_category: string;
  col_category: string;
}

interface SPIHeatmapData {
  figure: {
    title: string;
    panel_layout: { row_labels: string[]; col_labels: string[] };
  };
  data_rows: DataRow[];
}

const spiColor = (v: number) => {
  // Diverging: red (-1) -> white (0) -> blue (+1)
  if (v >= 0) {
    const t = Math.min(v, 1);
    const r = Math.round(255 - t * 200);
    const g = Math.round(255 - t * 180);
    const b = 255;
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const t = Math.min(Math.abs(v), 1);
    const r = 255;
    const g = Math.round(255 - t * 190);
    const b = Math.round(255 - t * 190);
    return `rgb(${r}, ${g}, ${b})`;
  }
};

export default function SPIHeatmap({ data }: { data: SPIHeatmapData }) {
  const panels = useMemo(() => {
    const panelMap = new Map<string, { title: string; rows: DataRow[] }>();
    data.data_rows.forEach((r) => {
      const key = `${r.panel_row}-${r.panel_col}`;
      if (!panelMap.has(key)) panelMap.set(key, { title: r.panel_title, rows: [] });
      panelMap.get(key)!.rows.push(r);
    });
    return [...panelMap.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [data]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {panels.map(([key, { title, rows }]) => {
        const models = [...new Set(rows.map((r) => r.row_category))];
        const datasets = [...new Set(rows.map((r) => r.col_category))];

        return (
          <div key={key}>
            <h5 className="font-sans text-xs font-semibold text-muted-foreground mb-2 text-center">
              {title}
            </h5>
            <div className="overflow-x-auto">
              <div className="inline-block min-w-[250px]">
                {/* Column headers */}
                <div className="flex ml-16">
                  {datasets.map((d) => (
                    <div
                      key={d}
                      className="flex-1 text-center font-sans text-[9px] font-medium text-muted-foreground truncate px-0.5"
                      title={d}
                    >
                      {d}
                    </div>
                  ))}
                </div>
                {/* Rows */}
                {models.map((m) => (
                  <div key={m} className="flex items-center">
                    <div className="w-16 text-right pr-1.5 font-sans text-[9px] font-medium text-muted-foreground truncate" title={m}>
                      {m}
                    </div>
                    {datasets.map((d) => {
                      const cell = rows.find(
                        (r) => r.row_category === m && r.col_category === d
                      );
                      const v = cell?.value ?? null;
                      return (
                        <div
                          key={d}
                          className="flex-1 aspect-square flex items-center justify-center text-[8px] font-mono border border-background"
                          style={{
                            backgroundColor: v !== null ? spiColor(v) : "hsl(var(--muted))",
                            color: v !== null && Math.abs(v) > 0.5 ? "#fff" : "#333",
                            minWidth: 32,
                            minHeight: 32,
                          }}
                          title={`${m} × ${d}: ${v?.toFixed(3) ?? "N/A"}`}
                        >
                          {v !== null ? v.toFixed(2) : "–"}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })}

      {/* Diverging color legend */}
      <div className="md:col-span-2 flex items-center justify-center gap-2 mt-2">
        <span className="font-sans text-[10px] text-muted-foreground">−1.0</span>
        <div
          className="w-40 h-3 rounded"
          style={{
            background: `linear-gradient(to right, ${spiColor(-1)}, ${spiColor(-0.5)}, ${spiColor(0)}, ${spiColor(0.5)}, ${spiColor(1)})`,
          }}
        />
        <span className="font-sans text-[10px] text-muted-foreground">+1.0</span>
        <span className="font-sans text-[10px] text-muted-foreground ml-1">SPI</span>
      </div>
    </div>
  );
}
