import { useMemo } from "react";
import { Tooltip as RechartsTooltip } from "recharts";

interface DataRow {
  panel_row: number;
  panel_col: number;
  panel_title: string;
  train_dataset: string;
  test_dataset: string;
  train_dataset_title: string;
  test_dataset_title: string;
  llm_name: string;
  probe_model: string;
  layer: number;
  metric: string;
  value: number;
  series_role: string;
}

interface TransferData {
  figure: {
    title: string;
    panel_layout: { row_labels: string[]; col_labels: string[] };
  };
  data_rows: DataRow[];
}

const COLOR_SCALE = (v: number, min: number, max: number) => {
  const t = max === min ? 0.5 : (v - min) / (max - min);
  // Blue (low) to orange (high)
  const r = Math.round(36 + t * (220 - 36));
  const g = Math.round(80 + (1 - Math.abs(t - 0.5) * 2) * 80);
  const b = Math.round(180 - t * 140);
  return `rgb(${r}, ${g}, ${b})`;
};

export default function TransferHeatmap({ data }: { data: TransferData }) {
  const { matrix, labels, minVal, maxVal } = useMemo(() => {
    const labels = data.figure.panel_layout.row_labels;
    const n = labels.length;

    // For each train->test pair, compute average minimum RMSE across layers for primary series
    const matrix: (number | null)[][] = Array.from({ length: n }, () =>
      Array(n).fill(null)
    );

    // Group by panel and compute min RMSE (best layer) averaged across models
    const panelData = new Map<string, number[]>();
    data.data_rows
      .filter((r) => r.series_role === "primary_series")
      .forEach((r) => {
        const key = `${r.panel_row}-${r.panel_col}`;
        if (!panelData.has(key)) panelData.set(key, []);
        panelData.get(key)!.push(r.value);
      });

    let minVal = Infinity,
      maxVal = -Infinity;
    panelData.forEach((values, key) => {
      const [row, col] = key.split("-").map(Number);
      const avg = values.reduce((a, b) => a + b, 0) / values.length;
      if (row < n && col < n) {
        matrix[row][col] = avg;
        minVal = Math.min(minVal, avg);
        maxVal = Math.max(maxVal, avg);
      }
    });

    return { matrix, labels, minVal, maxVal };
  }, [data]);

  return (
    <div className="w-full overflow-x-auto flex justify-center">
      <div className="inline-block min-w-[400px] pb-4">
        {/* Column labels */}
        <div className="flex ml-24">
          {labels.map((l) => (
            <div
              key={l}
              className="flex-1 text-center font-sans text-[10px] font-medium text-muted-foreground px-0.5 truncate"
              title={l}
            >
              {l}
            </div>
          ))}
        </div>

        {/* Matrix rows */}
        {matrix.map((row, ri) => (
          <div key={ri} className="flex items-center">
            <div className="w-24 text-right pr-2 font-sans text-[10px] font-medium text-muted-foreground truncate" title={labels[ri]}>
              {labels[ri]}
            </div>
            {row.map((val, ci) => (
              <div
                key={ci}
                className="flex-1 aspect-square flex items-center justify-center text-[9px] font-mono border border-background"
                style={{
                  backgroundColor:
                    val !== null
                      ? COLOR_SCALE(val, minVal, maxVal)
                      : "hsl(var(--muted))",
                  color: val !== null && val > (minVal + maxVal) / 2 ? "#fff" : "#333",
                  minWidth: 40,
                  minHeight: 40,
                }}
                title={`${labels[ri]} → ${labels[ci]}: ${val?.toFixed(3) ?? "N/A"}`}
              >
                {val !== null ? val.toFixed(2) : "–"}
              </div>
            ))}
          </div>
        ))}

        {/* Color scale legend */}
        <div className="flex items-center gap-2 mt-3 ml-24">
          <span className="font-sans text-[10px] text-muted-foreground">{minVal.toFixed(2)}</span>
          <div
            className="flex-1 h-3 rounded"
            style={{
              background: `linear-gradient(to right, ${COLOR_SCALE(minVal, minVal, maxVal)}, ${COLOR_SCALE((minVal + maxVal) / 2, minVal, maxVal)}, ${COLOR_SCALE(maxVal, minVal, maxVal)})`,
            }}
          />
          <span className="font-sans text-[10px] text-muted-foreground">{maxVal.toFixed(2)}</span>
          <span className="font-sans text-[10px] text-muted-foreground ml-1">RMSE</span>
        </div>
      </div>
    </div>
  );
}
