"use client";

import { CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import { Manrope, Space_Grotesk } from "next/font/google";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope" });
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
});

type Resort = {
  id: string;
  name: string;
  score: number;
  details: {
    new_snow_cm: number;
    rain_mm: number;
    temp_min: number;
    temp_max: number;
    gust_max: number;
    snow_depth_m: number | null;
  };
};

function cn(...xs: Array<string | false | undefined | null>) {
  return xs.filter(Boolean).join(" ");
}

function ArrowLeft({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path
        fill="currentColor"
        d="m10.5 5 1.4 1.4L7.3 11H20v2H7.3l4.6 4.6-1.4 1.4-7-7 7-7Z"
      />
    </svg>
  );
}

function ArrowRight({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path
        fill="currentColor"
        d="M13.5 5 12.1 6.4 16.7 11H4v2h12.7l-4.6 4.6L13.5 19l7-7-7-7Z"
      />
    </svg>
  );
}

export default function Home() {
  const [resorts, setResorts] = useState<Resort[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const cardRefs = useRef<Array<HTMLDivElement | null>>([]);

  useEffect(() => {
    const run = async () => {
      try {
        const base = process.env.NEXT_PUBLIC_API_URL;
        const res = await fetch(`${base}/rankings/tomorrow`);
        const json = await res.json();
        setResorts(json.resorts ?? []);
      } finally {
        setLoading(false);
      }
    };
    run();
  }, []);

  const sortedResorts = useMemo(() => {
    return [...resorts].sort((a, b) => b.score - a.score);
  }, [resorts]);

  const topResorts = useMemo(() => sortedResorts.slice(0, 6), [sortedResorts]);

  const resortCards = useMemo(() => sortedResorts, [sortedResorts]);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;

    let raf = 0;
    const update = () => {
      window.cancelAnimationFrame(raf);
      raf = window.requestAnimationFrame(() => {
        const cards = cardRefs.current.filter(Boolean) as HTMLElement[];
        if (cards.length === 0) return;
        const center = el.scrollLeft + el.clientWidth / 2;
        let closest = 0;
        let min = Number.POSITIVE_INFINITY;
        cards.forEach((child, idx) => {
          const childCenter = child.offsetLeft + child.offsetWidth / 2;
          const dist = Math.abs(center - childCenter);
          if (dist < min) {
            min = dist;
            closest = idx;
          }
        });
        setActiveIndex(closest);
      });
    };

    update();
    el.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.cancelAnimationFrame(raf);
      el.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, [resortCards.length]);

  const scrollToIndex = (nextIndex: number) => {
    const scroller = scrollerRef.current;
    const target = cardRefs.current[nextIndex];
    if (!scroller || !target) return;
    const offset = target.offsetLeft - (scroller.clientWidth - target.clientWidth) / 2;
    scroller.scrollTo({ left: offset, behavior: "smooth" });
  };

  return (
    <main
      className={`${manrope.variable} ${spaceGrotesk.variable} min-h-screen bg-[radial-gradient(circle_at_top,_#f5fbff_0%,_#e4f2ff_38%,_#cfe8ff_72%,_#bcdcff_100%)] text-slate-900`}
    >
      <div className="relative min-h-screen overflow-hidden">
        <div className="absolute -top-24 -left-24 h-64 w-64 rounded-full bg-white/70 blur-3xl" />
        <div className="absolute top-32 right-[-6rem] h-72 w-72 rounded-full bg-sky-200/60 blur-3xl" />
        <div className="absolute bottom-[-12rem] left-1/3 h-96 w-96 rounded-full bg-white/70 blur-[120px]" />

        <div className="relative mx-auto flex min-h-[80vh] max-w-6xl items-center px-6 pb-16 pt-20 lg:px-10">
          <div className="grid w-full gap-12 text-center lg:grid-cols-[1fr_1fr] lg:text-left">
            <div className="space-y-6 self-center">
              <div className="space-y-5">
                <h1 className="font-[var(--font-space-grotesk)] text-7xl font-bold leading-[1.02] tracking-tight text-slate-900 sm:text-8xl">
                  Shredderz
                </h1>
                <p className="mx-auto max-w-xl text-2xl text-slate-700 lg:mx-0">
                  Next-24h ski forecast for Canada's top resorts.
                </p>
              </div>
            </div>

            <div className="w-full rounded-3xl border border-white/70 bg-white/70 p-8 shadow-[0_18px_40px_rgba(30,60,100,0.12)]">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-500">
                    Resort Rankings
                  </p>
                  <p className="mt-2 font-[var(--font-space-grotesk)] text-3xl font-bold text-slate-900">
                    Movers and shakers
                  </p>
                </div>
                <button
                  className="rounded-full border border-slate-900/10 bg-white/80 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-700 shadow-sm hover:bg-white"
                  onClick={() => location.reload()}
                >
                  Refresh
                </button>
              </div>

              <div className="mt-6 divide-y divide-slate-900/20">
                {loading ? (
                  <div className="text-sm text-slate-500">Loading rankings…</div>
                ) : (
                  topResorts.map((resort, idx) => {
                    const isTop = idx < 3;

                    return (
                      <div
                        key={resort.id}
                        className={`py-4 ${
                          isTop ? "text-slate-900" : "text-slate-800"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div className="flex items-center gap-4">
                            <span className="text-sm font-semibold text-slate-500">
                              {String(idx + 1).padStart(2, "0")}
                            </span>
                            <p className={`text-lg ${isTop ? "font-semibold" : "font-medium"}`}>
                              {resort.name}
                            </p>
                          </div>
                          <div className="text-3xl font-semibold text-slate-900">
                            {resort.score.toFixed(1)}
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <section className="relative -mt-10 pb-24">
        <div className="mx-auto w-full max-w-none px-0">
          <div className="text-center">
            <h2 className="font-[var(--font-space-grotesk)] text-5xl font-bold tracking-tight text-slate-900 sm:text-6xl">
              Resort snow forecasts
            </h2>
            <div className="mt-6 flex items-center justify-center gap-6 text-2xl font-semibold text-slate-700">
              <button
                type="button"
                className="transition hover:text-slate-900 disabled:opacity-40"
                onClick={() => scrollToIndex(Math.max(activeIndex - 1, 0))}
                aria-label="Previous resort"
                disabled={activeIndex === 0}
              >
                <ArrowLeft className="h-6 w-6" />
              </button>
              <button
                type="button"
                className="transition hover:text-slate-900 disabled:opacity-40"
                onClick={() =>
                  scrollToIndex(Math.min(activeIndex + 1, resortCards.length - 1))
                }
                aria-label="Next resort"
                disabled={activeIndex === resortCards.length - 1}
              >
                <ArrowRight className="h-6 w-6" />
              </button>
            </div>
          </div>

          <style>{`.scrollbar-hidden::-webkit-scrollbar{display:none;}`}</style>
          <div
            className="relative mt-12 w-screen"
            style={
              {
                "--card-width": "min(62vw, 860px)",
                "--card-gap": "36px",
              } as CSSProperties
            }
          >
            <div
              ref={scrollerRef}
              className="scrollbar-hidden flex snap-x snap-mandatory gap-[var(--card-gap)] overflow-x-auto overflow-y-visible scroll-smooth pb-14 pt-8 [-ms-overflow-style:none] [scrollbar-width:none]"
              style={{
                paddingLeft: "calc((100vw - var(--card-width)) / 2)",
                paddingRight: "calc((100vw - var(--card-width)) / 2)",
              }}
            >
              {loading ? (
                <div className="text-sm text-slate-500">Loading resorts…</div>
              ) : resortCards.length === 0 ? (
                <div className="text-sm text-slate-500">No resorts found.</div>
              ) : (
                resortCards.map((resort, idx) => {
                  const distance = Math.abs(idx - activeIndex);
                  return (
                    <article
                      key={resort.id}
                      className="snap-center shrink-0 w-[var(--card-width)]"
                    >
                      <div
                        ref={(el) => {
                          cardRefs.current[idx] = el;
                        }}
                        onClick={() => {
                          if (idx !== activeIndex) scrollToIndex(idx);
                      }}
                        className={cn(
                          "relative h-[520px] cursor-pointer rounded-[36px] border border-white/80 bg-white/90 p-10 shadow-[0_24px_60px_rgba(30,60,100,0.14)] transition-all duration-700",
                          distance === 0 && "opacity-100 scale-[1.02]",
                          distance === 1 &&
                            "opacity-55 scale-[0.94] shadow-[0_18px_50px_rgba(60,100,140,0.15)]",
                          distance > 1 &&
                            "opacity-30 scale-[0.9] shadow-[0_12px_35px_rgba(60,100,140,0.12)]"
                        )}
                      >
                        <div className="absolute left-10 top-8 h-1.5 w-16 rounded-full bg-slate-900/10" />
                        <div className="mt-6 flex h-full flex-col justify-between">
                          <div className="min-h-[280px]">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-semibold text-slate-500">
                                #{String(idx + 1).padStart(2, "0")}
                              </span>
                              <span className="text-sm uppercase tracking-[0.3em] text-slate-500">
                                Tomorrow
                              </span>
                            </div>
                            <h3 className="mt-4 text-5xl font-bold tracking-tight text-slate-900">
                              {resort.name}
                            </h3>
                            <div className="mt-6 flex flex-wrap items-end gap-6 text-slate-700">
                              <div>
                                <p className="text-xs uppercase tracking-[0.3em] text-slate-500">
                                  Ski Score
                                </p>
                                <p className="mt-2 text-5xl font-semibold text-slate-900">
                                  {resort.score.toFixed(1)}
                                </p>
                              </div>
                            </div>
                            <div className="mt-8 flex flex-wrap items-start gap-x-12 gap-y-6 text-sm text-slate-600">
                              <div>
                                <p className="text-xs uppercase tracking-[0.28em] text-slate-500">
                                  New snow
                                </p>
                                <p className="mt-1 text-2xl font-semibold text-slate-900">
                                  {resort.details.new_snow_cm.toFixed(1)} cm
                                </p>
                              </div>
                              <div>
                                <p className="text-xs uppercase tracking-[0.28em] text-slate-500">
                                  Temp range
                                </p>
                                <p className="mt-1 text-2xl font-semibold text-slate-900">
                                  {resort.details.temp_min.toFixed(1)}° →{" "}
                                  {resort.details.temp_max.toFixed(1)}°
                                </p>
                              </div>
                              <div>
                                <p className="text-xs uppercase tracking-[0.28em] text-slate-500">
                                  Gusts
                                </p>
                                <p className="mt-1 text-2xl font-semibold text-slate-900">
                                  {resort.details.gust_max.toFixed(0)} km/h
                                </p>
                              </div>
                              <div>
                                <p className="text-xs uppercase tracking-[0.28em] text-slate-500">
                                  Rain
                                </p>
                                <p className="mt-1 text-2xl font-semibold text-slate-900">
                                  {resort.details.rain_mm.toFixed(1)} mm
                                </p>
                              </div>
                              <div>
                                <p className="text-xs uppercase tracking-[0.28em] text-slate-500">
                                  Base depth
                                </p>
                                <p className="mt-1 text-2xl font-semibold text-slate-900">
                                  {resort.details.snow_depth_m !== null
                                    ? `${resort.details.snow_depth_m.toFixed(1)} m`
                                    : "—"}
                                </p>
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center justify-between">
                            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-500">
                              {String(idx + 1).padStart(2, "0")} /{" "}
                              {String(resortCards.length).padStart(2, "0")}
                            </p>
                          </div>
                        </div>
                      </div>
                    </article>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
