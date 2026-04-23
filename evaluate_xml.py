"""
Evaluate YOLOPv2 against CVAT XML annotations (annotations.xml).

Supports both tasks:
  - Lane line    : polyline → rasterised mask, compared with lane_line_mask()
  - Driveable area: RLE mask → binary mask,  compared with driving_area_mask()

Usage
─────
  uv run evaluate_xml.py
  uv run evaluate_xml.py --xml annotations.xml --images test_images/test-real
"""

import sys
import csv
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

from utils.utils import time_synchronized, driving_area_mask, lane_line_mask, letterbox

XML_PATH   = ROOT / "annotations.xml"
IMAGE_DIR  = ROOT / "test_images" / "test-real"
MODEL_PATH = ROOT / "model" / "yolopv2.pt"
EVAL_DIR   = ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

WORK_SIZE = (1280, 720)
IMG_SIZE  = 640
STRIDE    = 32
LANE_THICKNESS = 48   # px — GT polyline width for both distance transform and comparison images
HIT_THRESHOLDS = [10, 20, 30, 50]   # px — distance thresholds for lane accuracy


# ── device ────────────────────────────────────────────────────────────────────

def auto_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── CVAT XML parsing ──────────────────────────────────────────────────────────

def decode_rle(rle_str: str, left: int, top: int, width: int, height: int,
               img_w: int, img_h: int) -> np.ndarray:
    """Decode CVAT RLE mask → full-image binary uint8 mask (0/255)."""
    counts = list(map(int, rle_str.split(",")))
    total  = width * height
    flat   = np.zeros(total, dtype=np.uint8)
    idx, val = 0, 0
    for run in counts:
        if val == 1:
            flat[idx: idx + run] = 255
        idx += run
        val ^= 1

    # RLE is column-major (Fortran order)
    patch = flat.reshape((height, width), order="F")

    full = np.zeros((img_h, img_w), dtype=np.uint8)
    y1, y2 = top, min(top + height, img_h)
    x1, x2 = left, min(left + width, img_w)
    full[y1:y2, x1:x2] = patch[: y2 - y1, : x2 - x1]
    return full


# ── Distance-based lane metric ────────────────────────────────────────────────

def sample_polyline_points(pts: list[tuple[int,int]], step_px: int = 5) -> np.ndarray:
    """Sample points every step_px along a polyline. Returns Nx2 array (x,y)."""
    sampled = []
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i + 1]
        n = max(1, int(np.hypot(x1-x0, y1-y0) / step_px))
        for j in range(n):
            t = j / n
            sampled.append((x0 + t*(x1-x0), y0 + t*(y1-y0)))
    sampled.append(pts[-1])
    return np.array(sampled, dtype=np.float32)


def lane_distance_metrics(pred_mask: np.ndarray,
                           polylines: list[list[tuple[int,int]]],
                           img_w: int, img_h: int,
                           thresholds: list[int] = HIT_THRESHOLDS,
                           step_px: int = 5) -> dict:
    """
    Full distance-based lane evaluation using distance transforms:

    GT→Pred direction (per sampled GT point):
      Recall@Npx   — % GT points within N px of prediction
      mean/median dist GT→Pred

    Pred→GT direction (per predicted pixel):
      Precision@Npx — % pred pixels within N px of GT polyline
      FPR@Npx       — % pred pixels farther than N px (false positives)

    Combined:
      F1@Npx        — harmonic mean of Recall@N and Precision@N
      Chamfer dist  — (mean GT→Pred + mean Pred→GT) / 2
      Hausdorff dist — max(max GT→Pred, max Pred→GT)
    """
    empty = ({f"recall@{t}px": 0.0 for t in thresholds} |
             {f"prec@{t}px":   0.0 for t in thresholds} |
             {f"f1@{t}px":     0.0 for t in thresholds} |
             {f"fpr@{t}px":    1.0 for t in thresholds} |
             {"chamfer": np.nan, "hausdorff": np.nan,
              "mean_gt2pred": np.nan, "mean_pred2gt": np.nan, "n_points": 0})

    # Rasterise GT lines for distance transform
    gt_thin = np.zeros((img_h, img_w), dtype=np.uint8)
    all_gt_pts = []
    for pl in polylines:
        if len(pl) < 2:
            continue
        pts_arr = np.array(pl, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(gt_thin, [pts_arr], False, 255, LANE_THICKNESS)
        pts = sample_polyline_points(pl, step_px)
        pts[:, 0] = pts[:, 0].clip(0, img_w - 1)
        pts[:, 1] = pts[:, 1].clip(0, img_h - 1)
        all_gt_pts.append(pts)

    if not all_gt_pts or gt_thin.max() == 0:
        return {**empty, "n_points": 0}

    gt_pts = np.vstack(all_gt_pts)   # (M,2) xy

    # Distance transforms (float32, L2)
    # pred_dt[y,x] = distance to nearest PREDICTED lane pixel
    # gt_dt[y,x]   = distance to nearest GT lane pixel
    pred_bg  = (255 - pred_mask).astype(np.uint8)
    gt_bg    = (255 - gt_thin).astype(np.uint8)
    pred_dt  = cv2.distanceTransform(pred_bg, cv2.DIST_L2, 5)
    gt_dt    = cv2.distanceTransform(gt_bg,   cv2.DIST_L2, 5)

    # GT→Pred: look up pred_dt at sampled GT point locations
    gy = gt_pts[:, 1].astype(int).clip(0, img_h-1)
    gx = gt_pts[:, 0].astype(int).clip(0, img_w-1)
    gt2pred = pred_dt[gy, gx]

    # Pred→GT: look up gt_dt at all predicted pixel locations
    pred_yx = np.argwhere(pred_mask > 0)
    if len(pred_yx) == 0:
        return {**empty, "n_points": len(gt_pts)}
    pred2gt = gt_dt[pred_yx[:, 0], pred_yx[:, 1]]

    result = {"n_points": len(gt_pts)}
    for t in thresholds:
        acc  = float((gt2pred  <= t).mean())
        prec = float((pred2gt  <= t).mean())
        f1   = 2*acc*prec/(acc+prec) if (acc+prec) > 0 else 0.0
        result[f"recall@{t}px"] = acc
        result[f"prec@{t}px"] = prec
        result[f"f1@{t}px"]   = f1
        result[f"fpr@{t}px"]  = 1.0 - prec

    result["mean_gt2pred"] = float(gt2pred.mean())
    result["mean_pred2gt"] = float(pred2gt.mean())
    result["chamfer"]      = float((gt2pred.mean() + pred2gt.mean()) / 2)
    result["hausdorff"]    = float(max(gt2pred.max(), pred2gt.max()))
    return result


def parse_xml(xml_path: Path) -> dict[str, dict]:
    """Return {filename: {lane_mask, driveable_mask}} using original image coords."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    result = {}

    for img_el in root.findall("image"):
        name   = img_el.get("name")        # e.g. test_01.jpg
        img_w  = int(img_el.get("width"))
        img_h  = int(img_el.get("height"))

        # ── lane polylines ────────────────────────────────────────────────
        polylines = []
        lane_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for el in img_el.findall("polyline"):
            if el.get("label") != "lane":
                continue
            pts_str = el.get("points", "")
            pts = []
            for pair in pts_str.split(";"):
                x_s, y_s = pair.split(",")
                pts.append((round(float(x_s)), round(float(y_s))))
            polylines.append(pts)
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(lane_mask, [pts_arr], isClosed=False,
                          color=255, thickness=LANE_THICKNESS)

        # ── driveable RLE mask ────────────────────────────────────────────
        da_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for el in img_el.findall("mask"):
            if el.get("label") != "driveable":
                continue
            rle   = el.get("rle")
            left  = int(el.get("left"))
            top   = int(el.get("top"))
            w     = int(el.get("width"))
            h     = int(el.get("height"))
            da_mask = decode_rle(rle, left, top, w, h, img_w, img_h)

        result[name] = {"lane": lane_mask, "polylines": polylines,
                        "driveable": da_mask, "w": img_w, "h": img_h}
    return result


# ── inference ─────────────────────────────────────────────────────────────────

def infer(model, device, half, orig: np.ndarray):
    im0w = cv2.resize(orig, WORK_SIZE, interpolation=cv2.INTER_LINEAR)
    lb, _, _ = letterbox(im0w, IMG_SIZE, stride=STRIDE)
    img_np = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))
    img = torch.from_numpy(img_np).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    img = img.unsqueeze(0)

    t1 = time_synchronized()
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(img)
    t2 = time_synchronized()

    h0, w0 = orig.shape[:2]

    da_raw = driving_area_mask(seg)
    da = cv2.resize(da_raw.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
    da = (da > 0).astype(np.uint8) * 255

    ll_raw = lane_line_mask(ll)
    ll_m = cv2.resize(ll_raw.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
    ll_m = (ll_m > 0).astype(np.uint8) * 255

    return da, ll_m, t2 - t1


# ── metrics ───────────────────────────────────────────────────────────────────

def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = (pred > 0).astype(bool)
    g = (gt   > 0).astype(bool)
    TP = int(( p &  g).sum())
    FP = int(( p & ~g).sum())
    FN = int((~p &  g).sum())
    TN = int((~p & ~g).sum())
    precision   = TP / (TP + FP)            if (TP + FP)   > 0 else 0.0
    recall      = TP / (TP + FN)            if (TP + FN)   > 0 else 0.0
    f1          = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    iou         = TP / (TP + FP + FN)       if (TP + FP + FN) > 0 else 0.0
    accuracy    = (TP + TN) / (TP + FP + FN + TN)
    fp_rate     = FP / (FP + TN)            if (FP + TN)   > 0 else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=precision, recall=recall, f1=f1, iou=iou,
                accuracy=accuracy, fp_rate=fp_rate)


# ── comparison image ──────────────────────────────────────────────────────────

def make_cmp(orig, gt, pred, label: str, out_path: Path):
    h, w = orig.shape[:2]
    scale = 800 / w
    th, tw = int(h * scale), 800

    def overlay(mask, color):
        vis = orig.copy()
        o = np.zeros_like(vis)
        o[mask > 0] = color
        return cv2.addWeighted(vis, 0.6, o, 0.4, 0)

    diff = orig.copy()
    p, g = (pred > 0), (gt > 0)
    diff[p &  g] = (0, 220, 0)
    diff[p & ~g] = (0, 0, 220)
    diff[~p & g] = (220, 0, 0)

    row = []
    for panel, lbl in [(overlay(gt, (0,200,0)), f"GT {label}"),
                       (overlay(pred, (0,100,255)), f"Pred {label}"),
                       (diff, "TP/FP/FN")]:
        r = cv2.resize(panel, (tw, th))
        cv2.putText(r, lbl, (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(r, lbl, (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
        row.append(r)
    cv2.imwrite(str(out_path), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ── report helpers ────────────────────────────────────────────────────────────

def _pct(v): return f"{v*100:.2f}%"

def print_da_section(rows: list[dict], times: list[float]) -> str:
    print(f"\n{'='*60}\n  Driveable Area\n{'='*60}")
    header = ["Image","Precision","Recall","F1","IoU","FP-Rate","Accuracy","ms"]
    sep    = [":---"] + ["---:"]*7
    lines  = ["| "+" | ".join(header)+" |", "| "+" | ".join(sep)+" |"]
    for r, t in zip(rows, times):
        lines.append(f"| {r['name']} | {_pct(r['precision'])} | {_pct(r['recall'])} | "
                     f"{_pct(r['f1'])} | {_pct(r['iou'])} | {_pct(r['fp_rate'])} | "
                     f"{_pct(r['accuracy'])} | {t*1000:.0f} |")
    def avg(k): return np.mean([r[k] for r in rows])
    t_ms = np.array(times)*1000
    lines.append(f"| **Mean** | **{_pct(avg('precision'))}** | **{_pct(avg('recall'))}** | "
                 f"**{_pct(avg('f1'))}** | **{_pct(avg('iou'))}** | **{_pct(avg('fp_rate'))}** | "
                 f"**{_pct(avg('accuracy'))}** | **{t_ms.mean():.0f}±{t_ms.std():.0f}** |")
    table = "\n".join(lines)
    print(table)
    return table


def print_lane_section(rows: list[dict], times: list[float]) -> str:
    print(f"\n{'='*72}\n  Lane Line — Distance-Based Metrics\n{'='*72}")
    def avg(k): return np.nanmean([r[k] for r in rows])
    t_ms = np.array(times) * 1000

    all_sections = []

    # ── Per-threshold tables ──────────────────────────────────────────
    for T in HIT_THRESHOLDS:
        print(f"\n  @ {T}px threshold")
        header = ["Image", f"Recall@{T}", f"Prec@{T}", f"F1@{T}", f"FPR@{T}"]
        sep    = [":---"] + ["---:"] * 4
        lines  = ["| "+" | ".join(header)+" |", "| "+" | ".join(sep)+" |"]
        for r in rows:
            lines.append(
                f"| {r['name']} "
                f"| {r[f'recall@{T}px']*100:.1f}% "
                f"| {r[f'prec@{T}px']*100:.1f}% "
                f"| {r[f'f1@{T}px']*100:.1f}% "
                f"| {r[f'fpr@{T}px']*100:.1f}% |"
            )
        lines.append(
            f"| **Mean** "
            f"| **{avg(f'recall@{T}px')*100:.1f}%** "
            f"| **{avg(f'prec@{T}px')*100:.1f}%** "
            f"| **{avg(f'f1@{T}px')*100:.1f}%** "
            f"| **{avg(f'fpr@{T}px')*100:.1f}%** |"
        )
        tbl = "\n".join(lines)
        print(tbl)
        all_sections.append(f"### @ {T}px threshold\n\n{tbl}")

    # ── Distance summary table ─────────────────────────────────────────
    print("\n  Distance Summary")
    header2 = ["Image", "Mean GT→Pred", "Mean Pred→GT", "Chamfer", "Hausdorff", "Points", "ms"]
    sep2    = [":---"] + ["---:"] * 6
    lines2  = ["| "+" | ".join(header2)+" |", "| "+" | ".join(sep2)+" |"]
    for r, t in zip(rows, times):
        lines2.append(
            f"| {r['name']} "
            f"| {r['mean_gt2pred']:.1f} px "
            f"| {r['mean_pred2gt']:.1f} px "
            f"| {r['chamfer']:.1f} px "
            f"| {r['hausdorff']:.1f} px "
            f"| {r['n_points']} "
            f"| {t*1000:.0f} |"
        )
    lines2.append(
        f"| **Mean** "
        f"| **{avg('mean_gt2pred'):.1f} px** "
        f"| **{avg('mean_pred2gt'):.1f} px** "
        f"| **{avg('chamfer'):.1f} px** "
        f"| **{avg('hausdorff'):.1f} px** "
        f"| — "
        f"| **{t_ms.mean():.0f}±{t_ms.std():.0f}** |"
    )
    tbl2 = "\n".join(lines2)
    print(tbl2)
    all_sections.append(f"### Distance Summary\n\n{tbl2}")
    return "\n\n".join(all_sections)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml",    default=str(XML_PATH))
    ap.add_argument("--images", default=str(IMAGE_DIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lane-thickness", type=int, default=LANE_THICKNESS,
                    help="Polyline brush width when rasterising lane GT (px)")
    args = ap.parse_args()

    xml_path  = Path(args.xml)
    img_dir   = Path(args.images)
    thickness = args.lane_thickness

    print(f"Parsing {xml_path.name} …")
    annotations = parse_xml(xml_path)
    print(f"  Found {len(annotations)} annotated images")

    device = auto_device(args.device)
    half   = device.type == "cuda"
    model  = torch.jit.load(str(MODEL_PATH), map_location=device).to(device).eval()
    if half: model.half()
    dummy = torch.zeros(1, 3, IMG_SIZE, int(IMG_SIZE * 0.6)).to(device)
    with torch.no_grad():
        try: model(dummy)
        except Exception: pass
    print(f"Model on {device}\n")

    da_rows, da_times = [], []
    ll_rows, ll_times = [], []

    for fname, ann in sorted(annotations.items()):
        # find image file
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / (Path(fname).stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"  ⚠ image not found for {fname}")
            continue

        orig = cv2.imread(str(img_path))
        gt_da   = ann["driveable"]
        gt_lane = ann["lane"]

        # resize GT if image was loaded at different size
        h0, w0 = orig.shape[:2]
        if gt_da.shape != (h0, w0):
            gt_da   = cv2.resize(gt_da,   (w0, h0), interpolation=cv2.INTER_NEAREST)
            gt_lane = cv2.resize(gt_lane, (w0, h0), interpolation=cv2.INTER_NEAREST)
        # re-rasterise lane at correct resolution if resized
        # (polylines were drawn at original resolution — thick enough)

        pred_da, pred_ll, inf_sec = infer(model, device, half, orig)

        m_da = pixel_metrics(pred_da, gt_da)
        m_da["name"] = Path(fname).stem
        da_rows.append(m_da)
        da_times.append(inf_sec)

        # Distance-based lane metric (scale polylines to current image size)
        scale_x = w0 / ann["w"]
        scale_y = h0 / ann["h"]
        scaled_pls = [[(round(x * scale_x), round(y * scale_y)) for x, y in pl]
                      for pl in ann["polylines"]]
        m_lane = lane_distance_metrics(pred_ll, scaled_pls, w0, h0)
        m_lane["name"] = Path(fname).stem
        ll_rows.append(m_lane)
        ll_times.append(inf_sec)

        print(f"[{fname}]")
        print(f"  DA   F1={m_da['f1']*100:.1f}%  IoU={m_da['iou']*100:.1f}%  "
              f"Prec={m_da['precision']*100:.1f}%  Rec={m_da['recall']*100:.1f}%")
        print(f"  Lane F1@20px={m_lane['f1@20px']*100:.1f}%  "
              f"Rec@20px={m_lane['recall@20px']*100:.1f}%  "
              f"Prec@20px={m_lane['prec@20px']*100:.1f}%  "
              f"Chamfer={m_lane['chamfer']:.1f}px  "
              f"Hausdorff={m_lane['hausdorff']:.1f}px  n={m_lane['n_points']}")

        make_cmp(orig, gt_da,   pred_da,  "DA",   EVAL_DIR / f"{Path(fname).stem}_xml_da.jpg")
        make_cmp(orig, gt_lane, pred_ll,  "Lane", EVAL_DIR / f"{Path(fname).stem}_xml_lane.jpg")

    if not da_rows:
        sys.exit("No results.")

    da_md   = print_da_section(da_rows, da_times)
    lane_md = print_lane_section(ll_rows, ll_times)

    # CSV — driveable
    da_csv = EVAL_DIR / "xml_da_report.csv"
    with open(da_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","precision","recall","f1","iou",
                                          "fp_rate","accuracy","TP","FP","FN","TN","inf_ms"],
                           extrasaction="ignore")
        w.writeheader()
        for r, t in zip(da_rows, da_times):
            w.writerow({**r, "inf_ms": round(t*1000, 2)})
    print(f"CSV → {da_csv}")

    # CSV — lane (distance-based)
    ll_csv = EVAL_DIR / "xml_lane_report.csv"
    lane_fields = (["name"] +
                   [f"{m}@{t}px" for t in HIT_THRESHOLDS for m in ("recall","prec","f1","fpr")] +
                   ["mean_gt2pred","mean_pred2gt","chamfer","hausdorff","n_points","inf_ms"])
    with open(ll_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lane_fields, extrasaction="ignore")
        w.writeheader()
        for r, t in zip(ll_rows, ll_times):
            w.writerow({**r, "inf_ms": round(t*1000, 2)})
    print(f"CSV → {ll_csv}")

    # Markdown
    md_path = EVAL_DIR / "xml_report.md"
    md_path.write_text(
        "# YOLOPv2 — CVAT XML Evaluation\n\n"
        f"- **Annotation:** {xml_path.name}\n"
        f"- **Images:** {len(da_rows)}\n"
        f"- **Lane GT thickness:** {thickness} px\n\n"
        "## Driveable Area\n\n" + da_md + "\n\n"
        "## Lane Line (Distance-Based)\n\n"
        f"> GT polyline thickness: {LANE_THICKNESS}px\n\n"
        "> **Recall@Npx** — % GT points within N px of prediction  \n"
        "> **Prec@Npx** — % predicted pixels within N px of GT  \n"
        "> **F1@Npx** — harmonic mean  \n"
        "> **FPR@Npx** — % predicted pixels farther than N px from GT\n\n"
        + lane_md + "\n"
    )
    print(f"MD  → {md_path}")


if __name__ == "__main__":
    main()
