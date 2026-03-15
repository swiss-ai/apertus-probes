import { ReactNode } from "react";
import { Download } from "lucide-react";
import { withBasePath } from "@/lib/paths";

interface FigureCardProps {
  id?: string;
  number: number;
  title: string;
  caption: string;
  interpretations?: string[];
  dataPath?: string;
  wide?: boolean;
  children: ReactNode;
}

export default function FigureCard({
  id,
  number,
  title,
  caption,
  interpretations,
  dataPath,
  wide,
  children,
}: FigureCardProps) {
  const handleDownload = () => {
    if (!dataPath) return;
    const resolvedDataPath = withBasePath(dataPath);
    const a = document.createElement("a");
    a.href = resolvedDataPath;
    a.download = dataPath.split("/").pop() || "data.json";
    a.click();
  };

  return (
    <div
      id={id}
      className={`figure-card ${wide ? "-mx-4 sm:-mx-6 md:-mx-12 lg:-mx-20 xl:-mx-28" : ""}`}
    >
      <div className="figure-label">Figure {number}</div>
      <h4 className="font-sans text-base font-semibold mb-3">{title}</h4>
      <div className="min-h-[200px]">{children}</div>
      <p className="figure-caption">{caption}</p>
      {interpretations && interpretations.length > 0 && (
        <div className="figure-interpretation">
          <ul className="list-disc ml-4 space-y-1">
            {interpretations.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
