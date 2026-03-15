import { Brain, Target, Zap, Sparkles, ArrowDown, XCircle, ArrowRight } from "lucide-react";

export default function ExplainerDiagram() {
  return (
    <div className="w-full max-w-4xl mx-auto my-12 p-6 md:p-12 bg-white rounded-[2.5rem] border border-slate-200/60 shadow-xl shadow-slate-200/40 font-sans relative overflow-hidden">

      {/* Background ambient lighting */}
      <div className="absolute top-0 right-0 -mt-20 -mr-20 w-80 h-80 bg-blue-400/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 -mb-20 -ml-20 w-80 h-80 bg-orange-400/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="text-center mb-12 relative z-10">
        <h3 className="text-2xl md:text-3xl font-extrabold tracking-tight text-slate-900 mb-3 flex items-center justify-center gap-3">
          Mechanistic Error Reduction
        </h3>
        <p className="text-slate-500 font-medium max-w-lg mx-auto text-sm md:text-base">
          Directly intervening on the model's <strong className="text-blue-600 font-semibold select-none">residual stream</strong> during generation to catch and correct mistakes.
        </p>
      </div>

      {/* Main Diagram Area */}
      <div className="relative w-full flex flex-col items-center">

        {/* Input */}
        <div className="bg-slate-900 text-white px-6 py-3 rounded-2xl shadow-xl shadow-slate-900/10 flex items-center gap-3 z-20 relative">
          <Sparkles size={16} className="text-blue-400" />
          <span className="font-mono text-xs sm:text-sm">"Who discovered penicillin?"</span>
        </div>

        <div className="h-6 w-px bg-slate-300 z-10" />

        {/* Central Core (3 columns) */}
        <div className="flex gap-4 sm:gap-8 justify-center items-stretch w-full">

          {/* Left Panel: Layers */}
          <div className="flex flex-col justify-center flex-1 items-end py-10 relative z-20">

            {/* Layer L */}
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-3 sm:p-4 w-36 sm:w-44 flex items-center gap-3 relative group transition">
              <div className="absolute top-1/2 -right-4 sm:-right-8 w-4 sm:w-8 h-px bg-slate-300">
                <div className="absolute top-1/2 right-0 w-1.5 h-1.5 rounded-full bg-slate-400 -translate-y-1/2 translate-x-1/2" />
              </div>
              <div className="bg-slate-50 p-2 rounded-xl text-slate-500 shadow-inner border border-slate-100">
                <Brain size={18} />
              </div>
              <span className="text-xs sm:text-sm font-bold text-slate-700">Layer L</span>
            </div>

          </div>

          {/* Center Panel: Residual Stream */}
          <div className="w-16 sm:w-24 relative flex flex-col items-center justify-center">
            <div className="w-full h-full bg-gradient-to-b from-blue-50/50 via-blue-100/40 to-blue-50/50 border border-blue-200/60 rounded-[2rem] flex flex-col items-center justify-center relative overflow-hidden shadow-inner backdrop-blur-md z-10 py-16">

              {/* Center track line */}
              <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-px bg-blue-300/40" />

              {/* Text */}
              <div className="flex flex-col items-center justify-center h-full">
                <span className="text-[10px] md:text-sm font-bold text-blue-900/40 uppercase tracking-[0.25em] writing-vertical-lr rotate-180 select-none bg-white/40 px-1 py-4 rounded-full backdrop-blur z-10 w-full text-center">
                  Residual Stream
                </span>
              </div>

              {/* Flow indicators */}
              <ArrowDown className="text-blue-300 absolute top-4 z-0" size={16} />
              <ArrowDown className="text-blue-300 absolute bottom-4 z-0" size={16} />
            </div>
          </div>

          {/* Right Panel: Interventions */}
          <div className="flex flex-col justify-center gap-6 flex-1 items-start py-6 sm:py-8 relative z-20">

            {/* Error Probe (Read) */}
            <div className="bg-orange-50 border border-orange-200 rounded-2xl shadow-sm p-3 sm:p-4 w-40 sm:w-52 flex items-center gap-3 relative z-20">
              <div className="absolute top-1/2 -left-4 sm:-left-8 w-4 sm:w-8 border-b-2 border-dashed border-orange-300">
                <div className="absolute top-1/2 right-0 w-2 h-2 border-t-2 border-r-2 border-orange-400 rotate-45 -translate-y-[calc(50%+1px)] translate-x-[-2px]" />
                <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] font-mono font-bold text-orange-950 bg-white/60 px-1 rounded whitespace-nowrap backdrop-blur">
                  read h
                </span>
              </div>
              <div className="bg-orange-500 p-2 rounded-xl text-white shadow-md shadow-orange-500/20">
                <Target size={18} />
              </div>
              <div>
                <h4 className="text-xs sm:text-sm font-bold text-orange-950 leading-none mb-1">Error Probe</h4>
                <p className="text-[9px] sm:text-[10px] text-orange-700/80 uppercase tracking-widest font-semibold">Predicts Error</p>
              </div>
            </div>

            {/* Steering Vector (Inject) - Same Layer Context */}
            <div className="bg-emerald-50 border border-emerald-200 rounded-2xl shadow-sm p-3 sm:p-4 w-40 sm:w-52 flex items-center gap-3 relative z-20">
              <div className="absolute top-1/2 -left-4 sm:-left-8 w-4 sm:w-8 border-b-2 border-dashed border-emerald-400">
                <div className="absolute top-1/2 left-0 w-2 h-2 border-b-2 border-l-2 border-emerald-500 rotate-45 -translate-y-[calc(50%-1px)] translate-x-[2px]" />
                <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] font-mono font-bold text-emerald-950 bg-white/60 px-1 rounded whitespace-nowrap backdrop-blur">
                  inject +v
                </span>
              </div>
              <div className="bg-emerald-500 p-2 rounded-xl text-white shadow-md shadow-emerald-500/20">
                <Zap size={18} />
              </div>
              <div>
                <h4 className="text-xs sm:text-sm font-bold text-emerald-950 leading-none mb-1">MERA Steering</h4>
                <p className="text-[9px] sm:text-[10px] text-emerald-700/80 uppercase tracking-widest font-semibold">Corrects State</p>
              </div>
            </div>

          </div>

        </div>

        <div className="h-6 w-px bg-slate-300 z-10" />

        {/* Output */}
        <div className="flex flex-col sm:flex-row gap-6 mt-2 items-center z-20 relative">

          {/* Wrong Answer (Base) */}
          <div className="bg-white border-2 border-red-500/20 text-slate-400 px-4 py-3 rounded-2xl shadow-sm flex items-center gap-3 relative opacity-60 line-through">
            <XCircle size={16} className="text-red-400" />
            <span className="font-mono text-xs sm:text-sm font-bold">C) Marie Curie</span>
            <span className="absolute -top-3 left-1/2 -translate-x-1/2 text-[9px] bg-red-50 text-red-600 px-1 rounded border border-red-100 font-bold tracking-wider not-italic normal-case no-underline whitespace-nowrap">Base Predicted</span>
          </div>

          <ArrowRight className="text-slate-300 hidden sm:block opacity-50" />

          {/* Correct Answer (MERA Steered) */}
          <div className="bg-white border-2 border-emerald-500/40 text-emerald-900 px-6 py-4 rounded-2xl shadow-lg flex items-center gap-3 relative">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-400 to-emerald-600 rounded-[1.1rem] blur opacity-20" />
            <div className="relative flex items-center gap-3">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.8)]" />
              <span className="font-mono text-xs sm:text-sm font-bold tracking-tight">A) Alexander Fleming</span>
            </div>
            <span className="absolute -top-3 left-1/2 -translate-x-1/2 text-[10px] bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded border border-emerald-200 font-bold tracking-wider uppercase whitespace-nowrap z-10 shadow-sm">MERA Steered</span>
          </div>

        </div>

      </div>

    </div>
  );
}
