import { useEffect, useState, ReactNode } from "react";

interface JsonChartLoaderProps<T> {
  path: string;
  validate?: (data: unknown) => data is T;
  children: (data: T) => ReactNode;
}

export default function JsonChartLoader<T>({
  path,
  validate,
  children,
}: JsonChartLoaderProps<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(path)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        // Handle NaN values in JSON (not valid JSON but common in Python output)
        const cleaned = text.replace(/:\s*NaN/g, ": null");
        const parsed = JSON.parse(cleaned);
        if (validate && !validate(parsed)) {
          throw new Error("Schema validation failed");
        }
        if (!cancelled) setData(parsed as T);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [path, validate]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground font-sans text-sm">
        Loading data…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-48 rounded-lg border-2 border-dashed border-border p-6 text-center">
        <p className="font-sans text-sm font-medium text-muted-foreground">
          Required data file not found
        </p>
        <p className="font-mono text-xs text-muted-foreground mt-1">{path}</p>
        {error && (
          <p className="font-sans text-xs text-destructive mt-2">
            Error: {error}
          </p>
        )}
      </div>
    );
  }

  return <>{children(data)}</>;
}
