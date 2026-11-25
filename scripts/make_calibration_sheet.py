#!/usr/bin/env python3
"""Generate an A4 calibration sheet for the DOFBOT wrist camera.

Two chessboards, both with dimensions printed on the sheet:

* A (close range, wrist camera): 7x5 squares of 11 mm  (inner corners 6x4)
* B (table / observer camera):  9x6 squares of 20 mm  (inner corners 8x5)

Plus a 100 mm ruler so the print scale can be verified before any calibration
run. Everything is laid out in millimetres on an exact A4 page.

    python scripts/make_calibration_sheet.py --out calib/a4_target.pdf

Print at 100% ("actual size"), never "fit to page".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A4_W_MM, A4_H_MM = 210.0, 297.0

# (name, square_mm, cols, rows, caption)
TARGET_WRIST = ("A", 11.0, 7, 5)
TARGET_TABLE = ("B", 20.0, 9, 6)


def draw_board(ax, x0, y0, square, cols, rows, invert_start=False) -> None:
    """Checkerboard with its lower-left corner at (x0, y0), in mm."""
    for r in range(rows):
        for c in range(cols):
            black = (r + c) % 2 == (0 if not invert_start else 1)
            if black:
                ax.add_patch(
                    Rectangle(
                        (x0 + c * square, y0 + r * square),
                        square,
                        square,
                        facecolor="black",
                        edgecolor="none",
                    )
                )
    # thin outline so the extent is unambiguous when cutting the sheet
    ax.add_patch(
        Rectangle(
            (x0, y0),
            cols * square,
            rows * square,
            facecolor="none",
            edgecolor="0.35",
            linewidth=0.4,
        )
    )


def draw_ruler(ax, x0, y0, length_mm: float = 100.0) -> None:
    ax.plot([x0, x0 + length_mm], [y0, y0], color="black", linewidth=0.6)
    for i in range(int(length_mm) + 1):
        if i % 10 == 0:
            h, w = 4.0, 0.6
        elif i % 5 == 0:
            h, w = 2.6, 0.5
        else:
            h, w = 1.4, 0.4
        ax.plot([x0 + i, x0 + i], [y0, y0 + h], color="black", linewidth=w)
    ax.text(x0, y0 - 2.4, "0", ha="center", va="top", fontsize=5)
    ax.text(
        x0 + length_mm,
        y0 - 2.4,
        f"{length_mm:.0f} mm",
        ha="center",
        va="top",
        fontsize=5,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="calib/a4_target.pdf")
    ap.add_argument("--png", default=None, help="also write a PNG preview here")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(A4_W_MM / 25.4, A4_H_MM / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, A4_W_MM)
    ax.set_ylim(0, A4_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        105,
        288,
        "DOFBOT SE - camera calibration sheet",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        105,
        281.5,
        "Print at 100% (actual size). Do NOT use 'fit to page'. "
        "Check the 100 mm ruler with a caliper before calibrating.",
        ha="center",
        va="center",
        fontsize=7.2,
    )
    ax.text(
        105,
        276.5,
        "Matte paper, flat on the desk. Keep the sheet clean and unbent: a warp of "
        "1 mm already biases the focal length.",
        ha="center",
        va="center",
        fontsize=6.6,
        color="0.25",
    )

    # --- target A: close range, fits the wrist camera at ~10-20 cm -----------
    _name, sq, cols, rows = TARGET_WRIST
    ax0, ay0 = 20.0, 205.0
    draw_board(ax, ax0, ay0, sq, cols, rows)
    ax.text(
        ax0,
        ay0 - 5.0,
        f"A  -  wrist camera (close range): {cols}x{rows} squares, {sq:.0f} mm  "
        f"->  {cols * sq:.0f} x {rows * sq:.0f} mm board, inner corners "
        f"{cols - 1}x{rows - 1}",
        ha="left",
        va="center",
        fontsize=6.8,
    )

    # --- target B: table / observer camera ----------------------------------
    _name, sq, cols, rows = TARGET_TABLE
    bx0, by0 = 15.0, 45.0
    draw_board(ax, bx0, by0, sq, cols, rows)
    ax.text(
        bx0,
        by0 - 5.0,
        f"B  -  table / observer camera: {cols}x{rows} squares, {sq:.0f} mm  "
        f"->  {cols * sq:.0f} x {rows * sq:.0f} mm board, inner corners {cols - 1}x{rows - 1}",
        ha="left",
        va="center",
        fontsize=6.8,
    )

    # --- scale ruler ---------------------------------------------------------
    draw_ruler(ax, 105.0, 20.0, 100.0)
    ax.text(
        105.0,
        30.0,
        "scale check: this line must measure exactly 100 mm (10 cm)",
        ha="left",
        va="center",
        fontsize=6.4,
    )

    ax.text(
        105,
        8,
        "geometry: page 210 x 297 mm, drawn in millimetres; squares are exact by construction",
        ha="center",
        va="center",
        fontsize=5.8,
        color="0.35",
    )

    fig.savefig(out, format="pdf")
    if args.png:
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
