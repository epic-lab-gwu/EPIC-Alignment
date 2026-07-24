from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images" / "promo"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

COLORS = {
    "ink": "#242832",
    "muted": "#5f6678",
    "line": "#d8d7d0",
    "paper": "#f7f5ef",
    "panel": "#ffffff",
    "dark": "#23262d",
    "teal": "#05998f",
    "red": "#e5584c",
    "green": "#449b5f",
    "blue": "#4770bc",
    "purple": "#825ab5",
    "gold": "#efb02f",
}


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    if bold:
        name += "-Bold"
    return ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = word if not current else f"{current} {word}"
            if text_size(draw, candidate, fnt)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_lines(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=outline or fill, width=2)


def grid(draw: ImageDraw.ImageDraw, width: int, height: int, step: int = 60) -> None:
    for x in range(0, width + 1, step):
        draw.line((x, 0, x, height), fill="#e7e4dc", width=1)
    for y in range(0, height + 1, step):
        draw.line((0, y, width, y), fill="#e7e4dc", width=1)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int = 8,
) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")
    r = width + 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def trajectory_points(
    x0: int,
    y0: int,
    w: int,
    h: int,
    phase: float,
    lift: float = 0.0,
    samples: int = 160,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in range(samples):
        t = i / (samples - 1)
        x = x0 + t * w
        y = y0 + h * (0.52 + 0.28 * math.sin(2.4 * math.pi * t + phase) + 0.08 * math.sin(6.2 * math.pi * t + phase))
        y += lift
        pts.append((x, y))
    return pts


def badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, fill: str) -> int:
    x, y = xy
    fnt = font(28, bold=True)
    tw, th = text_size(draw, label, fnt)
    pad_x, pad_y = 18, 9
    draw.rounded_rectangle((x, y, x + tw + pad_x * 2, y + th + pad_y * 2), radius=22, fill=fill)
    draw.text((x + pad_x, y + pad_y - 2), label, font=fnt, fill="#ffffff")
    return x + tw + pad_x * 2 + 12


def metric_row(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: str,
    label: str,
    value: str,
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, "#f6f6f2", "#d8d7d0")
    draw.rectangle((x1, y1, x1 + 12, y2), fill=color)
    draw.text((x1 + 34, y1 + 24), label, font=font(20, bold=True), fill=COLORS["muted"])
    vw, _ = text_size(draw, value, font(26, bold=True))
    draw.text((x2 - vw - 24, y1 + 20), value, font=font(26, bold=True), fill=COLORS["ink"])


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, optimize=True)


def hero() -> None:
    img = Image.new("RGB", (1600, 900), COLORS["paper"])
    draw = ImageDraw.Draw(img)
    grid(draw, 1600, 900)

    draw.text((150, 130), "EPICA Alignment", font=font(78, bold=True), fill=COLORS["ink"])
    draw_wrapped(
        draw,
        (154, 230),
        "Time-aware trajectory alignment and evaluation for visual-inertial systems, benchmarks, and OpenVINS workflows.",
        font(32),
        "#404656",
        800,
        10,
    )
    x = 154
    for label, color in [
        ("SE3", COLORS["teal"]),
        ("PosYaw", COLORS["blue"]),
        ("Sim3", COLORS["purple"]),
        ("APE/RPE", COLORS["red"]),
        ("OpenVINS", COLORS["dark"]),
    ]:
        x = badge(draw, (x, 380), label, color)

    gt = trajectory_points(150, 405, 850, 300, 0.5, 50)
    raw = trajectory_points(210, 390, 850, 290, 1.1, -18)
    aligned = trajectory_points(170, 405, 850, 300, 0.52, 25)
    draw_polyline(draw, raw, COLORS["red"], 8)
    draw_polyline(draw, gt, COLORS["teal"], 8)
    draw_polyline(draw, aligned, COLORS["green"], 8)

    lx = 160
    for label, color in [
        ("reference / GT", COLORS["teal"]),
        ("raw estimate", COLORS["red"]),
        ("EPICA aligned", COLORS["green"]),
    ]:
        draw.line((lx, 740, lx + 72, 740), fill=color, width=7)
        draw.ellipse((lx + 30, 731, lx + 48, 749), fill=color)
        draw.text((lx + 90, 724), label, font=font(26, bold=True), fill=COLORS["muted"])
        lx += 300

    rounded(draw, (1050, 150, 1490, 680), COLORS["panel"], COLORS["line"])
    draw.text((1110, 200), "Evaluation Report", font=font(38, bold=True), fill=COLORS["ink"])
    rows = [
        (COLORS["teal"], "ATE RMSE", "0.325 m"),
        (COLORS["blue"], "RPE 1s", "0.084 m"),
        (COLORS["green"], "Success Rate", "97.1%"),
        (COLORS["gold"], "Sim3 Scale", "1.0248"),
    ]
    y = 270
    for color, label, value in rows:
        metric_row(draw, (1110, y, 1440, y + 72), color, label, value)
        y += 92

    save(img, "epica_readme_hero.png")


def social_preview() -> None:
    img = Image.new("RGB", (1280, 640), COLORS["paper"])
    draw = ImageDraw.Draw(img)
    draw.polygon([(540, 0), (1280, 0), (1120, 640), (410, 640)], fill=COLORS["dark"])
    draw.polygon([(1100, 0), (1280, 0), (1280, 640), (1180, 640)], fill=COLORS["gold"])

    draw.text((70, 88), "EPICA", font=font(74, bold=True), fill=COLORS["ink"])
    draw.text((72, 178), "Alignment", font=font(48, bold=True), fill=COLORS["ink"])
    draw_wrapped(draw, (74, 262), "Reliable trajectory alignment for VIO evaluation, dataset benchmarks, and OpenVINS-compatible reports.", font(24), "#4d5361", 430, 8)

    pts = trajectory_points(565, 190, 330, 170, 0.0, 0, 70)
    draw_polyline(draw, pts, COLORS["teal"], 5)
    draw_polyline(draw, trajectory_points(575, 174, 350, 170, 0.7, -8, 70), COLORS["red"], 5)
    draw_polyline(draw, trajectory_points(565, 205, 345, 150, 0.25, 8, 70), COLORS["blue"], 4)

    rounded(draw, (915, 88, 1210, 432), COLORS["panel"], COLORS["dark"])
    draw.text((948, 118), "pip install epica", font=font(20, bold=True, mono=True), fill=COLORS["dark"])
    features = [
        (COLORS["teal"], "Time sync"),
        (COLORS["red"], "Extrinsics"),
        (COLORS["green"], "World alignment"),
        (COLORS["blue"], "APE / RPE / SR"),
        (COLORS["purple"], "OpenVINS compat"),
    ]
    y = 180
    for color, label in features:
        draw.ellipse((948, y + 5, 966, y + 23), fill=color)
        draw.text((982, y), label, font=font(19, bold=True), fill=COLORS["ink"])
        y += 48

    save(img, "epica_github_social_preview.png")


def pipeline() -> None:
    img = Image.new("RGB", (1600, 900), COLORS["paper"])
    draw = ImageDraw.Draw(img)
    grid(draw, 1600, 900)
    draw.text((112, 110), "From raw trajectories to trustworthy evaluation", font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((116, 184), "EPICA estimates time offset, solves alignment, then reports metrics and reliability signals.", font=font(22), fill=COLORS["muted"])

    stages = [
        ("1", "Time sync", "cross-correlation", COLORS["teal"]),
        ("2", "Extrinsics", "rotation + translation", COLORS["red"]),
        ("3", "World align", "SE3 / PosYaw / Sim3", COLORS["green"]),
        ("4", "Evaluate", "APE, RPE, SR", COLORS["blue"]),
    ]
    x = 120
    y = 365
    for idx, title, subtitle, color in stages:
        rounded(draw, (x, y, x + 275, y + 170), COLORS["panel"], COLORS["line"])
        draw.ellipse((x + 24, y + 24, x + 58, y + 58), fill=color)
        draw.text((x + 35, y + 27), idx, font=font(18, bold=True), fill="#ffffff")
        draw.text((x + 24, y + 76), title, font=font(28, bold=True), fill=COLORS["ink"])
        draw.text((x + 24, y + 118), subtitle, font=font(17), fill=COLORS["muted"])
        pts = trajectory_points(x + 22, y + 120, 220, 35, idx.count("1") * 0.3 + len(title), 10, 35)
        draw.line(pts, fill=color, width=4)
        if x < 1000:
            draw.line((x + 302, y + 85, x + 355, y + 85), fill=COLORS["ink"], width=3)
            draw.polygon([(x + 355, y + 85), (x + 342, y + 77), (x + 342, y + 93)], fill=COLORS["ink"])
        x += 365

    rounded(draw, (150, 665, 1110, 775), COLORS["dark"], COLORS["dark"])
    draw.text((190, 695), "$ epa gt.tum estimate.tum --mode sim3", font=font(27, bold=True, mono=True), fill="#ffffff")
    draw.text((190, 735), "outputs: metrics.json | plots | report.html | interactive_report.html", font=font(18, mono=True), fill="#c8cbd3")

    save(img, "epica_pipeline_overview.png")


def openvins() -> None:
    img = Image.new("RGB", (1200, 900), "#eef2ef")
    draw = ImageDraw.Draw(img)
    draw.text((70, 64), "OpenVINS-compatible outside,", font=font(50, bold=True), fill=COLORS["ink"])
    draw.text((70, 124), "EPA alignment inside.", font=font(50, bold=True), fill=COLORS["ink"])
    draw.text((72, 200), "Legacy ov_eval commands, robust timestamp handling, and EPICA metrics.", font=font(25), fill=COLORS["muted"])

    rounded(draw, (70, 300, 690, 760), COLORS["dark"], COLORS["dark"])
    for i, color in enumerate([COLORS["red"], COLORS["gold"], COLORS["green"]]):
        draw.ellipse((110 + i * 32, 335, 128 + i * 32, 353), fill=color)
    terminal = [
        "python -m epa.ov_eval_compat \\",
        "  error_comparison \\",
        "  GT/aqualoc/archaeo \\",
        "  benchmark/archaeo/pose",
        "",
        "TOOL SOURCE: epa_step3=0, epa_eval=20",
        "SR distance = 97.06% | time = 97.12%",
        "Sim3 scale = 1.024849 | reliable = true",
    ]
    yy = 385
    for line in terminal:
        draw.text((105, yy), line, font=font(21, mono=True), fill="#f7f7f1")
        yy += 35

    cards = [
        (COLORS["teal"], "Strict when it matches", "Preserves ov_eval-style timestamp association."),
        (COLORS["red"], "Fallback when needed", "Resamples sparse cases onto the GT timeline."),
        (COLORS["blue"], "Batch-safe reports", "--fail-on-skipped supports CI and releases."),
    ]
    y = 320
    for color, title, subtitle in cards:
        rounded(draw, (760, y, 1130, y + 120), COLORS["panel"], COLORS["line"])
        draw.rectangle((760, y, 776, y + 120), fill=color)
        draw.text((802, y + 28), title, font=font(28, bold=True), fill=COLORS["ink"])
        draw_wrapped(draw, (802, y + 70), subtitle, font(19), COLORS["muted"], 285, 6)
        y += 160

    save(img, "epica_openvins_compat.png")


def metrics() -> None:
    img = Image.new("RGB", (1200, 900), COLORS["paper"])
    draw = ImageDraw.Draw(img)
    draw.text((72, 70), "Metrics that explain the run,", font=font(50, bold=True), fill=COLORS["ink"])
    draw.text((72, 130), "not just score it.", font=font(50, bold=True), fill=COLORS["ink"])
    draw.text((74, 205), "APE/RPE, drift validity, success-rate gates, and Sim3 reliability.", font=font(25), fill=COLORS["muted"])

    rounded(draw, (80, 295, 720, 760), COLORS["panel"], COLORS["line"])
    draw.text((125, 335), "APE translation over distance", font=font(28, bold=True), fill=COLORS["ink"])
    origin = (130, 690)
    draw.line((origin[0], origin[1], 665, origin[1]), fill="#969da9", width=3)
    draw.line((origin[0], origin[1], origin[0], 420), fill="#969da9", width=3)
    draw.text((130, 714), "distance", font=font(18), fill=COLORS["muted"])
    draw.text((92, 420), "error", font=font(18), fill=COLORS["muted"])
    draw.line((130, 545, 665, 545), fill=COLORS["green"], width=4)
    draw.text((510, 520), "global gate", font=font(20, bold=True), fill=COLORS["green"])
    curve: list[tuple[float, float]] = []
    for i in range(120):
        t = i / 119
        x = 140 + t * 510
        y = 580 - 90 * math.sin(math.pi * t) + 55 * math.sin(5 * math.pi * t + 0.4) - 20 * math.cos(12 * math.pi * t)
        curve.append((x, y))
    draw.line(curve, fill=COLORS["red"], width=5, joint="curve")
    for x, y in [curve[0], curve[68], curve[-1]]:
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=COLORS["red"])

    stats = [
        (COLORS["teal"], "APE RMSE", "0.325 m"),
        (COLORS["blue"], "RPE 1s", "0.084 m"),
        (COLORS["green"], "Drift-valid SR", "97.1%"),
        (COLORS["gold"], "Valid segments", "24 / 24"),
        (COLORS["purple"], "Sim3 reliable", "true"),
    ]
    y = 305
    for color, label, value in stats:
        rounded(draw, (770, y, 1120, y + 80), COLORS["panel"], COLORS["line"])
        draw.ellipse((800, y + 24, 832, y + 56), fill=color)
        draw.text((850, y + 14), label, font=font(20, bold=True), fill=COLORS["muted"])
        draw.text((850, y + 42), value, font=font(25, bold=True), fill=COLORS["ink"])
        y += 95

    save(img, "epica_metrics_reliability.png")


def benchmark() -> None:
    img = Image.new("RGB", (1200, 900), "#f2f6f5")
    draw = ImageDraw.Draw(img)
    draw.text((72, 78), "Benchmark many trajectories", font=font(45, bold=True), fill=COLORS["ink"])
    draw.text((72, 132), "without losing the details.", font=font(45, bold=True), fill=COLORS["ink"])
    draw.text((74, 204), "Run multi-dataset evaluation, inspect suspicious cases, and publish reproducible summaries.", font=font(22), fill=COLORS["muted"])

    rounded(draw, (100, 310, 1080, 760), COLORS["panel"], COLORS["line"])
    draw.text((150, 355), "summary_public.csv", font=font(24, bold=True, mono=True), fill=COLORS["ink"])
    draw.text((150, 395), "paper_tables/   cases/   logs/   epa_runs/", font=font(20, mono=True), fill=COLORS["muted"])

    labels = ["Aqualoc", "EuRoC", "Grand Tour", "LaMAR", "UZH FPV"]
    vals = [42, 76, 58, 85, 91]
    cols = [COLORS["teal"], COLORS["blue"], COLORS["red"], COLORS["purple"], COLORS["green"]]
    x = 150
    for label, val, color in zip(labels, vals, cols):
        bar_h = int(val * 2.3)
        draw.rounded_rectangle((x, 670 - bar_h, x + 78, 670), radius=10, fill=color)
        draw.text((x + 6, 690), label, font=font(15, bold=True), fill=COLORS["ink"])
        draw.text((x + 14, 682 - bar_h), f"{val}%", font=font(17, bold=True), fill="#ffffff")
        x += 160
    draw.line((135, 670, 970, 670), fill="#a8adb7", width=3)

    ribbons = [
        ("rank methods", COLORS["teal"]),
        ("spot failures", COLORS["red"]),
        ("compare datasets", COLORS["blue"]),
        ("export plots", COLORS["green"]),
    ]
    x = 155
    for label, color in ribbons:
        draw.rounded_rectangle((x, 418, x + 170, 446), radius=14, fill=color)
        draw.text((x + 14, 422), label, font=font(14, bold=True), fill="#ffffff")
        x += 190

    save(img, "epica_benchmark_suite.png")


def contact_sheet() -> None:
    names = [
        "epica_readme_hero.png",
        "epica_github_social_preview.png",
        "epica_pipeline_overview.png",
        "epica_openvins_compat.png",
        "epica_metrics_reliability.png",
        "epica_benchmark_suite.png",
    ]
    thumb_w, thumb_h = 420, 236
    sheet = Image.new("RGB", (900, 900), COLORS["paper"])
    draw = ImageDraw.Draw(sheet)
    positions = [(35, 35), (465, 35), (35, 318), (465, 318), (35, 601), (465, 601)]
    for name, (x, y) in zip(names, positions):
        im = Image.open(OUT / name)
        im.thumbnail((thumb_w, thumb_h))
        sheet.paste(im, (x, y))
        draw.text((x, y + thumb_h + 12), name, font=font(18, mono=True), fill=COLORS["ink"])
    save(sheet, "contact_sheet.png")


def main() -> None:
    hero()
    social_preview()
    pipeline()
    openvins()
    metrics()
    benchmark()
    contact_sheet()


if __name__ == "__main__":
    main()
