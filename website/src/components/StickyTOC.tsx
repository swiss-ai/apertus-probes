import { useEffect, useState } from "react";

interface TOCItem {
  id: string;
  label: string;
  level: number;
}

const tocItems: TOCItem[] = [
  { id: "abstract", label: "Abstract", level: 1 },
  { id: "motivation", label: "Motivation", level: 1 },
  { id: "contributions", label: "Contributions", level: 1 },
  { id: "setup", label: "Problem Setup", level: 1 },
  { id: "method", label: "Method", level: 1 },
  { id: "probes", label: "Probe Training", level: 2 },
  { id: "steering-methods", label: "Steering Methods", level: 2 },
  { id: "results", label: "Results", level: 1 },
  { id: "probe-results", label: "Probe Results", level: 2 },
  { id: "steering-results", label: "Steering Results", level: 2 },
  { id: "robustness", label: "Robustness & Ablations", level: 1 },
  { id: "limitations", label: "Limitations", level: 1 },
  { id: "conclusion", label: "Conclusion", level: 1 },
  { id: "code", label: "Code Availability", level: 1 },
];

export default function StickyTOC() {
  const [active, setActive] = useState("");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      { rootMargin: "-80px 0px -70% 0px", threshold: 0 }
    );

    tocItems.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    setOpen(false);
  };

  return (
    <>
      {/* Mobile toggle */}
      <button
        onClick={() => setOpen(!open)}
        className="lg:hidden fixed bottom-4 right-4 z-40 font-sans text-sm font-medium px-4 py-2 rounded-full shadow-lg border bg-background"
        aria-label="Table of contents"
      >
        {open ? "✕ Close" : "☰ Contents"}
      </button>

      {/* Mobile overlay */}
      {open && (
        <div
          className="lg:hidden fixed inset-0 z-30 bg-foreground/20 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        />
      )}

      <nav
        className={`
          fixed z-40 bg-background border-r
          lg:top-16 lg:left-0 lg:w-56 xl:w-64 lg:h-[calc(100vh-4rem)] lg:block lg:overflow-y-auto lg:py-6 lg:px-2
          ${open ? "bottom-0 left-0 right-0 rounded-t-2xl shadow-2xl p-6 max-h-[70vh] overflow-y-auto" : "hidden lg:block"}
        `}
      >
        <div className="font-sans text-xs font-semibold uppercase tracking-wider mb-3 px-3 text-muted-foreground">
          Contents
        </div>
        {tocItems.map((item) => (
          <button
            key={item.id}
            onClick={() => scrollTo(item.id)}
            className={`toc-link w-full text-left ${active === item.id ? "toc-link-active" : ""}`}
            style={{ paddingLeft: `${item.level === 2 ? 1.5 : 0.75}rem` }}
          >
            {item.label}
          </button>
        ))}
      </nav>
    </>
  );
}
