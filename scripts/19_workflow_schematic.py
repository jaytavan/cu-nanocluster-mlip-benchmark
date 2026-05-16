"""
19_workflow_schematic.py

Publication-Quality Workflow Schematic (Figure 1)
--------------------------------------------------
Three-panel horizontal layout showing the full benchmark workflow:
  Panel A — Dataset: DFT reference data generation
  Panel B — Zero-Shot Benchmark: zero-shot MLIP performance
  Panel C — Fine-Tuned Screening: fine-tuning improves recall

Uses only matplotlib.patches — no special fonts or external dependencies.

Output:
  viz/fig_workflow.png  (300 dpi, 7 × 3.5 inches)

Usage:
    python3 scripts/19_workflow_schematic.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
VIZ_DIR     = PROJECT_DIR / "viz"
VIZ_DIR.mkdir(exist_ok=True)

# ── Color scheme ──────────────────────────────────────────────────────────────
C_TENSORNET  = "#2196F3"   # blue
C_MACE       = "#FF6B35"   # orange
C_CHGNET     = "#4CAF50"   # green
C_NAVY       = "#1a1a2e"   # DFT/GPAW
C_COPPER     = "#b87333"   # accent / connecting arrows

# Panel background tints
BG_A = "#dceeff"   # light blue  — Dataset
BG_B = "#ffe8dc"   # light orange — Zero-shot
BG_C = "#dcf5e0"   # light green  — Fine-tuned

# Text colours
TC_TITLE  = "#1a1a2e"
TC_BODY   = "#2a2a3e"
TC_BADGE  = "#ffffff"
TC_WARN   = "#c0392b"

# Box fill colours
BC_STEP   = "#ffffff"    # step boxes
BC_OUT    = "#f5f5ff"    # output / result badges
BC_WARN   = "#fdecea"    # warning box


# ── Helper drawing functions ───────────────────────────────────────────────────

def fancy_box(ax, x, y, w, h, text, fontsize=8, fc=BC_STEP, ec="#333333",
              tc=TC_BODY, bold=False, style="round,pad=0.08", valign="center",
              text_wrap_width=None):
    """Draw a rounded-corner box with centred text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va=valign,
        fontsize=fontsize,
        fontweight=weight,
        color=tc,
        zorder=3,
        wrap=True,
        multialignment="center",
    )


def badge(ax, x, y, w, h, text, fc=C_NAVY, fontsize=7.5):
    """Coloured badge box (no rounded corners)."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=fc,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=TC_BADGE,
        zorder=4,
        multialignment="center",
    )


def arrow_down(ax, x, y_top, y_bot, color=C_COPPER, lw=1.5):
    """Vertical arrow pointing downward."""
    ax.annotate(
        "",
        xy=(x, y_bot),
        xytext=(x, y_top),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
        ),
        zorder=2,
    )


def arrow_right(ax, x_left, x_right, y, color=C_COPPER, lw=2.5):
    """Horizontal arrow pointing right."""
    ax.annotate(
        "",
        xy=(x_right, y),
        xytext=(x_left, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
        ),
        zorder=5,
    )


def panel_bg(ax, x, y, w, h, fc, label, label_color="#333333"):
    """Draw panel background rectangle and panel label."""
    rect = mpatches.Rectangle(
        (x, y), w, h,
        facecolor=fc,
        edgecolor="#aaaaaa",
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(rect)
    # Panel letter (A/B/C) in top-left
    ax.text(
        x + 0.02, y + h - 0.03, label,
        ha="left", va="top",
        fontsize=13,
        fontweight="bold",
        color=label_color,
        zorder=4,
    )


# ── Main drawing ──────────────────────────────────────────────────────────────

def draw_workflow():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Panel widths and positions (evenly centred, full margins) ─────────────
    pw = 3.4    # panel content width
    ph = 4.3    # panel height
    py = 0.35   # panel y start
    gap = 0.4   # gap between panels (arrow zone)

    x_a = 0.45
    x_b = x_a + pw + gap + 0.55    # 0.45 + 3.4 + 0.4 + 0.55 = 4.80
    x_c = x_b + pw + gap + 0.55    # 4.80 + 3.4 + 0.4 + 0.55 = 9.15
    # Panel C right edge = 9.15 + 3.4 + 0.1 = 12.65  →  fits in 13 ✓

    # Background panels
    panel_bg(ax, x_a - 0.1, py - 0.1, pw + 0.2, ph + 0.2, BG_A, "A")
    panel_bg(ax, x_b - 0.1, py - 0.1, pw + 0.2, ph + 0.2, BG_B, "B")
    panel_bg(ax, x_c - 0.1, py - 0.1, pw + 0.2, ph + 0.2, BG_C, "C")

    # ── Panel titles ──────────────────────────────────────────────────────────
    for (x_start, title, color) in [
        (x_a, "Dataset", C_NAVY),
        (x_b, "Zero-Shot Benchmark", TC_WARN),
        (x_c, "Fine-Tuned Screening", "#2e7d32"),
    ]:
        ax.text(
            x_start + pw / 2, py + ph - 0.02, title,
            ha="center", va="top",
            fontsize=10, fontweight="bold", color=color, zorder=4,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL A — Dataset
    # ═══════════════════════════════════════════════════════════════════════════
    bw, bh = 2.8, 0.44   # standard box dims
    bx = x_a + (pw - bw) / 2  # centred in panel

    y_a1 = py + ph - 0.85
    fancy_box(ax, bx, y_a1, bw, bh,
              "Cu nanoclusters\nCu₁₀ – Cu₅₀",
              fontsize=8.5, fc="#e8f4ff", ec=C_NAVY, tc=C_NAVY, bold=True)

    arrow_down(ax, bx + bw / 2, y_a1, y_a1 - 0.15, color=C_NAVY)

    y_a2 = y_a1 - 0.15 - bh
    fancy_box(ax, bx, y_a2, bw, bh,
              "H* site enumeration\n(atop / bridge / hollow)",
              fontsize=8, fc=BC_STEP, ec="#666666")

    arrow_down(ax, bx + bw / 2, y_a2, y_a2 - 0.15, color=C_NAVY)

    y_a3 = y_a2 - 0.15 - bh
    fancy_box(ax, bx, y_a3, bw, bh,
              "GPAW  PBE-D3(BJ)\nΔG$_{H*}$ = E$_{cluster+H}$ − E$_{cluster}$ − ½E$_{H_2}$",
              fontsize=7.5, fc=BC_STEP, ec="#666666")

    arrow_down(ax, bx + bw / 2, y_a3, y_a3 - 0.18, color=C_NAVY)

    # Output badge
    badge(ax,
          bx, y_a3 - 0.18 - 0.52,
          bw, 0.52,
          "364 DFT configurations\n3 isomers × 5 sizes",
          fc=C_NAVY, fontsize=8.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL B — Zero-Shot Benchmark
    # ═══════════════════════════════════════════════════════════════════════════
    bw_b = 2.8
    bx_b = x_b + (pw - bw_b) / 2

    # Three model boxes
    model_info = [
        ("MACE-MP-0",  C_MACE,      "MAE = 0.181 eV\nrecall₁₀ = 0%"),
        ("CHGNet",     C_CHGNET,    "MAE = 0.406 eV\nrecall₁₀ = 0%"),
        ("TensorNet",  C_TENSORNET, "MAE = 0.087 eV\nrecall₁₀ = 11%"),
    ]
    y_start_b = py + ph - 0.85
    box_h_b   = 0.56
    gap_b     = 0.12

    for i, (name, color, stats_text) in enumerate(model_info):
        yt = y_start_b - i * (box_h_b + gap_b)
        # Model name box (coloured header strip)
        header_h = 0.22
        fancy_box(ax, bx_b, yt + box_h_b - header_h, bw_b, header_h,
                  name, fontsize=8.5, fc=color, ec=color, tc="white", bold=True)
        # Stats body
        fancy_box(ax, bx_b, yt, bw_b, box_h_b - header_h,
                  stats_text, fontsize=8, fc=BC_STEP, ec=color, tc=TC_BODY,
                  style="round,pad=0.05")

    # Warning box
    y_warn = y_start_b - 3 * (box_h_b + gap_b) - 0.05
    warn_h = 0.52
    fancy_box(ax, bx_b, y_warn, bw_b, warn_h,
              "Zero-shot insufficient\nfor accurate HER screening",
              fontsize=8.5, fc=BC_WARN, ec=TC_WARN, tc=TC_WARN, bold=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL C — Fine-Tuned Screening
    # ═══════════════════════════════════════════════════════════════════════════
    bw_c = 2.8
    bx_c = x_c + (pw - bw_c) / 2

    y_c1 = py + ph - 0.85
    bh_c = 0.44

    fancy_box(ax, bx_c, y_c1, bw_c, bh_c,
              "N = 10 – 100 DFT\nfine-tuning points",
              fontsize=8.5, fc="#e8f5e9", ec="#2e7d32", tc="#2e7d32", bold=False)

    arrow_down(ax, bx_c + bw_c / 2, y_c1, y_c1 - 0.14, color="#2e7d32")

    y_c2 = y_c1 - 0.14 - bh_c
    fancy_box(ax, bx_c, y_c2, bw_c, bh_c,
              "Fine-tune MACE-MP-0",
              fontsize=9.5, fc=C_MACE, ec=C_MACE, tc="white", bold=True)

    arrow_down(ax, bx_c + bw_c / 2, y_c2, y_c2 - 0.14, color="#2e7d32")

    y_c3 = y_c2 - 0.14 - 0.58
    fancy_box(ax, bx_c, y_c3, bw_c, 0.58,
              "recall₁₀ = 21% at N=10 → 42% at N=40\nSpearman ρ = 0.70 at N=100",
              fontsize=8.5, fc=BC_STEP, ec="#2e7d32", tc="#2e7d32")

    arrow_down(ax, bx_c + bw_c / 2, y_c3, y_c3 - 0.14, color="#2e7d32")

    y_c4 = y_c3 - 0.14 - bh_c
    fancy_box(ax, bx_c, y_c4, bw_c, bh_c,
              "Screen all sites  (0.06 s/site GPU)\n→ submit top 20% to DFT",
              fontsize=8.5, fc=BC_STEP, ec="#666666")

    arrow_down(ax, bx_c + bw_c / 2, y_c4, y_c4 - 0.16, color="#2e7d32")

    # Savings badge
    badge(ax,
          bx_c, y_c4 - 0.16 - 0.46,
          bw_c, 0.46,
          "~5× DFT cost reduction",
          fc="#2e7d32", fontsize=9.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # Connecting arrows A → B and B → C
    # ═══════════════════════════════════════════════════════════════════════════
    # Arrow A → B  (centred vertically on panel)
    mid_y = py + ph / 2
    x_ab_left  = x_a + pw + 0.1
    x_ab_right = x_b - 0.1
    arrow_right(ax, x_ab_left, x_ab_right, mid_y, color=C_COPPER, lw=2.5)

    # Arrow B → C
    x_bc_left  = x_b + pw + 0.1
    x_bc_right = x_c - 0.1
    arrow_right(ax, x_bc_left, x_bc_right, mid_y, color=C_COPPER, lw=2.5)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "MLIP Benchmark Workflow for Cu Nanocluster HER Screening",
        fontsize=11, fontweight="bold", y=0.99, color=TC_TITLE,
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating workflow schematic ...")
    fig = draw_workflow()
    out_path = VIZ_DIR / "fig_workflow.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved: {out_path}  ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
