import { ReactNode } from "react";
import ReadingProgress from "./ReadingProgress";
import StickyTOC from "./StickyTOC";

export default function PaperLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <ReadingProgress />
      <StickyTOC />
      <main className="lg:ml-56 xl:ml-64">
        <div className="mx-auto max-w-article px-4 sm:px-6 py-12 md:py-20">
          {children}
        </div>
      </main>
    </div>
  );
}
