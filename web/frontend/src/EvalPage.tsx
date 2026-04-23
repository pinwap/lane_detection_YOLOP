import { useRef } from "react";
import html2canvas from "html2canvas";
import "./EvalPage.css";

// ── Data ─────────────────────────────────────────────────────────────────────

const CVAT_ROWS = [
  { name: "test_01", precision: 97.03, recall: 92.28, f1: 94.60, iou: 89.75, fpr: 0.39, acc: 98.73, ms: 279.6 },
  { name: "test_02", precision: 98.90, recall: 86.16, f1: 92.09, iou: 85.35, fpr: 0.24, acc: 97.07, ms: 61.5  },
  { name: "test_03", precision: 84.00, recall: 94.14, f1: 88.78, iou: 79.83, fpr: 3.07, acc: 96.52, ms: 63.0  },
  { name: "test_04", precision: 98.89, recall: 83.82, f1: 90.73, iou: 83.04, fpr: 0.40, acc: 94.85, ms: 81.1  },
  { name: "test_05", precision: 97.21, recall: 90.88, f1: 93.94, iou: 88.57, fpr: 0.47, acc: 98.22, ms: 66.3  },
];

const CVAT_MEAN = { precision: 95.21, recall: 89.46, f1: 92.03, iou: 85.31, fpr: 0.91, acc: 97.08, ms: 110.3 };

const LANE_ROWS = [
  { name: "test_01", recall: 79.2, prec: 94.5, f1: 86.2, fpr: 5.5 },
  { name: "test_02", recall: 96.8, prec: 52.6, f1: 68.2, fpr: 47.4 },
  { name: "test_03", recall: 97.3, prec: 50.9, f1: 66.8, fpr: 49.1 },
  { name: "test_04", recall: 100.0, prec: 56.8, f1: 72.4, fpr: 43.2 },
  { name: "test_05", recall: 99.3, prec: 75.2, f1: 85.6, fpr: 24.8 },
];

const LANE_MEAN = { recall: 94.5, prec: 66.0, f1: 75.8, fpr: 34.0 };

const TUSIMPLE_ROWS = [
  { method: "LaneNet (2018)",       acc: 96.4,  note: "Lane detection only" },
  { method: "SCNN (2018)",          acc: 96.5,  note: "Lane detection only" },
  { method: "YOLOPv2 (ours)",       acc: 96.18, note: "Detection + Seg + Lane", highlight: true },
  { method: "CLRNet (2022)",        acc: 97.1,  note: "Lane detection only" },
];

const TUSIMPLE_STATS = [
  { label: "TuSimple Acc",   value: "96.18%", sub: "±4.93%" },
  { label: "Recall",         value: "85.59%", sub: "±8.80%" },
  { label: "Precision",      value: "50.27%", sub: "±7.93%" },
  { label: "F1 Score",       value: "62.94%", sub: "±7.33%" },
  { label: "IoU",            value: "46.33%", sub: "±7.72%" },
  { label: "FP-Rate",        value: "2.35%",  sub: "±0.66%" },
  { label: "Pixel Accuracy", value: "97.33%", sub: "±0.69%" },
  { label: "Inf. Time (MPS)","value": "36.8 ms", sub: "~27 FPS" },
];

// ── PNG export via html2canvas ────────────────────────────────────────────────

async function downloadPNG(tableRef: React.RefObject<HTMLDivElement | null>, filename: string) {
  const el = tableRef.current;
  if (!el) return;
  const canvas = await html2canvas(el, {
    scale: 2,
    useCORS: true,
    backgroundColor: "#f4fbfe",
    ignoreElements: (element) => element.hasAttribute("data-export-ignore"),
  });
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, "image/png");
}

// ── Component ────────────────────────────────────────────────────────────────

export default function EvalPage() {
  const cvatTableRef = useRef<HTMLDivElement>(null);
  const laneTableRef = useRef<HTMLDivElement>(null);
  const tusimpleTableRef = useRef<HTMLDivElement>(null);

  const downloadTable = (ref: React.RefObject<HTMLDivElement | null>, filename: string) => {
    void downloadPNG(ref, filename);
  };

  return (
    <div className="eval-page">
      {/* ── Header strip ─────────────────────────────────────────────── */}
      <div className="eval-hero">
        <div className="eval-hero-inner">
          <div>
            <h1 className="eval-title">Evaluation Results</h1>
            <p className="eval-subtitle">YOLOPv2 · Driveable Area &amp; Lane Line Detection</p>
          </div>
        </div>
      </div>

      <div className="eval-body">

        {/* ── Section A: CVAT ──────────────────────────────────────────── */}
        <section className="eval-section">
          <div className="eval-section-head">
            <span className="eval-tag">CVAT</span>
            <div>
              <h2 className="eval-section-title">Driveable Area — Manual Annotation</h2>
              <p className="eval-section-sub">5 real-world images · annotated with CVAT · Apple MPS inference</p>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {/* Stat cards */}
          <div className="stat-cards">
            {[
              { label: "F1 Score",   value: `${CVAT_MEAN.f1.toFixed(2)}%`,  accent: true },
              { label: "IoU",        value: `${CVAT_MEAN.iou.toFixed(2)}%`,  accent: true },
              { label: "Precision",  value: `${CVAT_MEAN.precision.toFixed(2)}%` },
              { label: "Recall",     value: `${CVAT_MEAN.recall.toFixed(2)}%` },
              { label: "FP-Rate",    value: `${CVAT_MEAN.fpr.toFixed(2)}%` },
              { label: "Accuracy",   value: `${CVAT_MEAN.acc.toFixed(2)}%` },
            ].map((s) => (
              <div key={s.label} className={`stat-card${s.accent ? " stat-card--accent" : ""}`}>
                <span className="stat-label">{s.label}</span>
                <span className="stat-value">{s.value}</span>
              </div>
            ))}
          </div>

          {/* Per-image table */}
          <div className="tbl-wrap" ref={cvatTableRef}>
            <div className="tbl-headbar">
              <p className="tbl-caption">Per-image CVAT evaluation table</p>
              <button
                className="tbl-dl-btn"
                data-export-ignore="true"
                onClick={() => downloadTable(cvatTableRef, "yolopv2_cvat_table.png")}
              >
                Download
              </button>
            </div>
            <table className="eval-tbl">
              <thead>
                <tr>
                  {["Image","Precision","Recall","F1","IoU","FP-Rate","Accuracy","Inf (ms)"].map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {CVAT_ROWS.map((r) => (
                  <tr key={r.name}>
                    <td className="tbl-name">{r.name}</td>
                    <td>{r.precision.toFixed(2)}%</td>
                    <td>{r.recall.toFixed(2)}%</td>
                    <td>{r.f1.toFixed(2)}%</td>
                    <td>{r.iou.toFixed(2)}%</td>
                    <td>{r.fpr.toFixed(2)}%</td>
                    <td>{r.acc.toFixed(2)}%</td>
                    <td className="tbl-mono">{r.ms.toFixed(1)}</td>
                  </tr>
                ))}
                <tr className="tbl-mean">
                  <td className="tbl-name">Mean</td>
                  <td>{CVAT_MEAN.precision.toFixed(2)}%</td>
                  <td>{CVAT_MEAN.recall.toFixed(2)}%</td>
                  <td>{CVAT_MEAN.f1.toFixed(2)}%</td>
                  <td>{CVAT_MEAN.iou.toFixed(2)}%</td>
                  <td>{CVAT_MEAN.fpr.toFixed(2)}%</td>
                  <td>{CVAT_MEAN.acc.toFixed(2)}%</td>
                  <td className="tbl-mono">{CVAT_MEAN.ms.toFixed(1)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          </div>
        </section>

        {/* ── Section B: Lane Detection ─────────────────────────────── */}
        <section className="eval-section">
          <div className="eval-section-head">
            <span className="eval-tag eval-tag--purple">Lane Detection</span>
            <div>
              <h2 className="eval-section-title">Lane Detection — CVAT XML Evaluation</h2>
              <p className="eval-section-sub">5 real-world images · Recall@20, Precision@20, F1@20, and FPR@20</p>
            </div>
          </div>

          <div className="tbl-wrap" ref={laneTableRef}>
            <div className="tbl-headbar">
              <p className="tbl-caption">Lane detection table</p>
              <button
                className="tbl-dl-btn"
                data-export-ignore="true"
                onClick={() => downloadTable(laneTableRef, "yolopv2_lane_detection_table.png")}
              >
                Download
              </button>
            </div>
            <table className="eval-tbl">
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Image</th>
                  <th>Recall@20</th>
                  <th>Prec@20</th>
                  <th>F1@20</th>
                  <th>FPR@20</th>
                </tr>
              </thead>
              <tbody>
                {LANE_ROWS.map((r) => (
                  <tr key={r.name}>
                    <td className="tbl-name">{r.name}</td>
                    <td>{r.recall.toFixed(1)}%</td>
                    <td>{r.prec.toFixed(1)}%</td>
                    <td>{r.f1.toFixed(1)}%</td>
                    <td>{r.fpr.toFixed(1)}%</td>
                  </tr>
                ))}
                <tr className="tbl-mean">
                  <td className="tbl-name">Mean</td>
                  <td>{LANE_MEAN.recall.toFixed(1)}%</td>
                  <td>{LANE_MEAN.prec.toFixed(1)}%</td>
                  <td>{LANE_MEAN.f1.toFixed(1)}%</td>
                  <td>{LANE_MEAN.fpr.toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* ── Section C: TuSimple ──────────────────────────────────────── */}
        <section className="eval-section">
          <div className="eval-section-head">
            <span className="eval-tag eval-tag--purple">TuSimple</span>
            <div>
              <h2 className="eval-section-title">Lane Line — TuSimple Benchmark</h2>
              <p className="eval-section-sub">300 images from TuSimple test set · ±20 px threshold</p>
            </div>
          </div>

          {/* Stat grid */}
          <div className="stat-cards stat-cards--8">
            {TUSIMPLE_STATS.map((s) => (
              <div key={s.label} className={`stat-card${s.label === "TuSimple Acc" ? " stat-card--accent" : ""}`}>
                <span className="stat-label">{s.label}</span>
                <span className="stat-value">{s.value}</span>
                <span className="stat-sub">{s.sub}</span>
              </div>
            ))}
          </div>

          {/* SOTA comparison */}
          <div className="tbl-wrap" ref={tusimpleTableRef}>
            <div className="tbl-headbar">
              <p className="tbl-caption">State-of-the-art comparison on TuSimple</p>
              <button
                className="tbl-dl-btn"
                data-export-ignore="true"
                onClick={() => downloadTable(tusimpleTableRef, "yolopv2_tusimple_table.png")}
              >
                Download
              </button>
            </div>
            <table className="eval-tbl">
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Method</th>
                  <th>TuSimple Acc</th>
                  <th style={{ textAlign: "left" }}>Note</th>
                </tr>
              </thead>
              <tbody>
                {TUSIMPLE_ROWS.map((r) => (
                  <tr key={r.method} className={r.highlight ? "tbl-mean" : ""}>
                    <td className="tbl-name">{r.method}</td>
                    <td className={r.highlight ? "tbl-highlight" : ""}>{r.acc.toFixed(2)}%</td>
                    <td style={{ color: "var(--text-3)", fontSize: "12px" }}>{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="eval-note">
            <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.4"/>
              <path d="M8 7v4M8 5.5v.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
            </svg>
            <span>
              YOLOPv2 achieves <strong>96.18%</strong> TuSimple accuracy — comparable to single-task lane detectors,
              while simultaneously running vehicle detection and driveable area segmentation in one forward pass.
              Low Precision/IoU reflects thicker model predictions vs. the 10 px GT polylines used by TuSimple.
            </span>
          </div>
        </section>

      </div>{/* end eval-body */}
    </div>
  );
}
