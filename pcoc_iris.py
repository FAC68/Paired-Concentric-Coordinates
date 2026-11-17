#!/usr/bin/env python3
"""
Concentric Coordinates (CoC) vs. Paired Concentric Coordinates (PCoC)

Implements Algorithm 1 (A,B -> F) and Algorithm 2 (F -> A,B) from
section 7.4 of the attached document and applies them to the Fisher iris data.

Usage
-----
    python pcoc_iris.py --csv fisher_iris.csv --out pcoc_vs_coc.png
    python pcoc_iris.py --csv fisher_iris.csv --no-collocate --highlight 100

Conventions (the document does not fix these, so they are parameters)
---------------------------------------------------------------------
* Every attribute is min-max normalised to [0, 1].
* On its circle, value x sits at angle  theta = start + x * span
  (counter-clockwise from the +x axis).  Default span = 2*pi.
  NOTE: with span = 2*pi, x=0 and x=1 land on the same point; use e.g.
  --span 5.5 (radians) if you need them to be distinguishable.
* Circle k has radius k+1  (so the example in Fig. 35 uses radii 1 and 2).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

TAU = 2 * np.pi


# ----------------------------------------------------------------------------
# Basic CoC geometry
# ----------------------------------------------------------------------------
def minmax(X):
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def value_to_point(x, radius, span=TAU, start=0.0):
    """Normalised value(s) -> point(s) on a circle of given radius. Shape (N,2)."""
    th = start + np.asarray(x, float) * span
    return radius * np.stack([np.cos(th), np.sin(th)], axis=-1)


def point_to_value(P, radius, span=TAU, start=0.0):
    """Inverse of value_to_point."""
    th = (np.arctan2(P[..., 1], P[..., 0]) - start) % TAU
    return th / span


def attribute_radii(n, collocate):
    """Radius of the circle used by each of the n attributes.

    collocate=False : one circle per attribute  -> radii 1, 2, ..., n
    collocate=True  : X2=X3, X4=X5, ...         -> 4 attrs use 3 circles,
                      10 attrs use 6 circles (as in the document).
    """
    j = np.arange(n)
    circle_idx = (j + 1) // 2 if collocate else j
    return (circle_idx + 1).astype(float)


# ----------------------------------------------------------------------------
# Algorithm 1: A, B -> F          Algorithm 2: F -> A, B
# ----------------------------------------------------------------------------
def pair_forward(A, B, r1, r2):
    """Algorithm 1.  A on circle X1 (radius r1), B on circle X2 (radius r2).
    A, B: (N,2) arrays.  Returns F (N,2) and the side flags (sA, sB) needed
    for an exact inverse (see pair_inverse)."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    # 1. segment AB;  2-3. foot D of the perpendicular from the centre C=(0,0)
    t = B - A
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)  # unit normal of line AB
    s = np.sum(A * n, axis=1)
    n = np.where(s[:, None] < 0, -n, n)            # orient n from C towards the line
    d = np.abs(s)                                  # |CD|
    u = n                                          # unit vector C -> D
    # 4. G: extend CD to circle X2.   5. E: same ray meets circle X1
    G = r2 * u
    E = r1 * u
    # 6-7. r = |CG|/|CD|,  F = E + (G-E)/r      (1/r = d / r2: no division by 0)
    F = E + (G - E) * (d / r2)[:, None]
    # which side of D (along the line) A and B lie on
    tau = np.stack([-u[:, 1], u[:, 0]], axis=1)
    sA = np.sign(np.sum(A * tau, axis=1))
    sB = np.sign(np.sum(B * tau, axis=1))
    return F, (sA, sB)


def pair_inverse(F, r1, r2, signs=(-1, +1)):
    """Algorithm 2.  F -> (A, B).

    F fixes the line AB (its normal direction and distance d from C), but a
    line meets each circle in TWO points, so which one is A and which is B
    is not determined by F.  `signs=(sA, sB)` (each +-1, or an array) picks
    the side of D along the line.  The document's example (Fig. 35) uses
    (-1, +1): A and B on opposite sides of D.  Pass the flags returned by
    pair_forward for an exact round trip.
    """
    F = np.atleast_2d(F)
    sA, sB = signs
    cf = np.linalg.norm(F, axis=1)
    u = F / cf[:, None]                                 # 1. line CF
    G = r2 * u                                          # 2. G on X2
    # 3-4. E on X1 ;  1/r = |EF| / |EG|
    inv_r = np.clip((cf - r1) / (r2 - r1), 0, 1)
    D = G * inv_r[:, None]                              # 5. D = C + (G-C)/r
    d = np.linalg.norm(D, axis=1)
    tau = np.stack([-u[:, 1], u[:, 0]], axis=1)         # 6. line through D, orthogonal to CG
    hA = np.sqrt(np.maximum(r1**2 - d**2, 0))           # 7. intersections with X1, X2
    hB = np.sqrt(np.maximum(r2**2 - d**2, 0))
    A = D + (np.asarray(sA)[..., None] * hA[:, None]) * tau
    B = D + (np.asarray(sB)[..., None] * hB[:, None]) * tau
    return A, B


# ----------------------------------------------------------------------------
# Whole-dataset transforms
# ----------------------------------------------------------------------------
def coc_points(Xn, radii, span, start=0.0):
    """CoC: one point per attribute.  Returns (N, n, 2)."""
    return np.stack([value_to_point(Xn[:, j], radii[j], span, start)
                     for j in range(Xn.shape[1])], axis=1)


def pcoc_points(Xn, radii, span, start=0.0):
    """PCoC: one point per consecutive attribute pair (a leftover odd
    attribute is drawn as an ordinary CoC point).
    Returns points (N, m, 2) and the list of side flags per pair."""
    n = Xn.shape[1]
    pts, flags = [], []
    for j in range(0, n - 1, 2):
        A = value_to_point(Xn[:, j], radii[j], span, start)
        B = value_to_point(Xn[:, j + 1], radii[j + 1], span, start)
        F, sg = pair_forward(A, B, radii[j], radii[j + 1])
        pts.append(F)
        flags.append(sg)
    if n % 2:
        pts.append(value_to_point(Xn[:, -1], radii[-1], span, start))
    return np.stack(pts, axis=1), flags


def pcoc_roundtrip(Xn, radii, span, start=0.0):
    """Check F -> (A,B) -> values.  Returns (max error with flags kept,
    fraction of cases recovered with the fixed (-1,+1) convention)."""
    pts, flags = pcoc_points(Xn, radii, span, start)
    err, ok = 0.0, np.ones(len(Xn), bool)
    for k, j in enumerate(range(0, Xn.shape[1] - 1, 2)):
        r1, r2 = radii[j], radii[j + 1]
        for signs, track in ((flags[k], True), ((-1, +1), False)):
            A, B = pair_inverse(pts[:, k], r1, r2, signs)
            xa = point_to_value(A, r1, span, start)
            xb = point_to_value(B, r2, span, start)
            e = np.maximum(np.abs(xa - Xn[:, j]), np.abs(xb - Xn[:, j + 1]))
            e = np.minimum(e, 1 - e)   # values 0 and 1 coincide when span = 2*pi
            if track:
                err = max(err, e.max())
            else:
                ok &= e < 1e-6
    return err, ok.mean()


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
CLASS_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]


def draw_circles(ax, radii, labels):
    for r, lab in zip(radii, labels):
        ax.add_patch(Circle((0, 0), r, fill=False, ec="0.55", lw=1))
        ax.text(0, r, lab, ha="center", va="center", fontsize=8, color="0.2",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=5)
    ax.plot(0, 0, "k+", ms=6)
    m = max(radii) * 1.08
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_cases(ax, pts, y_codes, classes, highlight=None, arrows=False, dots=True):
    N, m, _ = pts.shape
    colors = [CLASS_COLORS[c % len(CLASS_COLORS)] for c in y_codes]
    segs = [pts[i, k:k + 2] for i in range(N) for k in range(m - 1)]
    segcol = [colors[i] for i in range(N) for _ in range(m - 1)]
    ax.add_collection(LineCollection(segs, colors=segcol, linewidths=0.7, alpha=0.5, zorder=2))
    if arrows:
        for (p, q), c in zip(segs, segcol):
            ax.annotate("", xy=q, xytext=p, zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7, alpha=0.6,
                                        mutation_scale=9, shrinkA=0, shrinkB=0))
    if dots:
        for c_i in np.unique(y_codes):
            P = pts[y_codes == c_i].reshape(-1, 2)
            ax.scatter(P[:, 0], P[:, 1], s=7, color=CLASS_COLORS[c_i % len(CLASS_COLORS)],
                       alpha=0.7, zorder=3, label=classes[c_i])
    if highlight is not None:
        P = pts[highlight]
        ax.plot(P[:, 0], P[:, 1], "-", color="gold", lw=6, alpha=0.9, zorder=6)
        ax.plot(P[:, 0], P[:, 1], "-o", color="red", lw=1.8, ms=5, zorder=7)


def circle_labels(names, collocate):
    n = len(names)
    if not collocate:
        return list(names)
    labs = [names[0]]
    for j in range(1, n, 2):
        labs.append(" / ".join(names[j:j + 2]))
    return labs


def plot_comparison(Xn, y_codes, classes, names, collocate, span, highlight, out, show):
    n = Xn.shape[1]
    r_coc = attribute_radii(n, collocate=False)
    r_pcoc = attribute_radii(n, collocate=collocate)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))

    coc = coc_points(Xn, r_coc, span)
    draw_circles(axes[0], sorted(set(r_coc)), circle_labels(names, False))
    draw_cases(axes[0], coc, y_codes, classes, highlight)
    axes[0].set_title(f"Concentric Coordinates (CoC)\n{n} points and {n-1} segments per case, "
                      f"{len(set(r_coc))} circles")

    pc, _ = pcoc_points(Xn, r_pcoc, span)
    draw_circles(axes[1], sorted(set(r_pcoc)), circle_labels(names, collocate))
    draw_cases(axes[1], pc, y_codes, classes, highlight, arrows=True)
    m = pc.shape[1]
    axes[1].set_title(f"Paired Concentric Coordinates (PCoC{', collocated' if collocate else ''})\n"
                      f"{m} points and {m-1} segment(s) per case, {len(set(r_pcoc))} circles")

    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(classes), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")
    return fig


def plot_construction(Xn_row, names, span, out):
    """Scaffolding view of Algorithm 1 for the first attribute pair of one case
    (analogous to Fig. 32a / Fig. 35)."""
    r1, r2 = 1.0, 2.0
    A = value_to_point(np.array([Xn_row[0]]), r1, span)
    B = value_to_point(np.array([Xn_row[1]]), r2, span)
    F, _ = pair_forward(A, B, r1, r2)
    A, B, F = A[0], B[0], F[0]
    u = F / np.linalg.norm(F)
    # D is recovered from F via Algorithm 2 geometry
    d = r2 * (np.linalg.norm(F) - r1) / (r2 - r1)
    D, G, E = d * u, r2 * u, r1 * u

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    draw_circles(ax, [r1, r2], [f"X1: {names[0]}", f"X2: {names[1]}"])
    ax.plot(*zip(A, B), "-", color="magenta", lw=2, label="segment AB (CoC)")
    ax.plot(*zip((0, 0), G), "--", color="gray", lw=1)
    ax.plot(*zip(E, G), "-", color="green", lw=3, alpha=0.5, label="segment EG")
    for P, name, c in ((A, "A", "magenta"), (B, "B", "magenta"), (D, "D", "k"),
                       (E, "E", "gray"), (G, "G", "gray"), (F, "F", "green")):
        ax.plot(*P, "o", color=c, ms=7, zorder=8)
        ax.annotate(name, P, xytext=(6, 6), textcoords="offset points", fontsize=11, zorder=9)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("PCoC construction (Algorithm 1): A, B -> F")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"saved {out}")
    return fig


# ----------------------------------------------------------------------------
# Self-test: reproduces the numeric example of Fig. 35
# ----------------------------------------------------------------------------
def selftest():
    F, _ = pair_forward(np.array([[1.0, 0.0]]), np.array([[0.0, 2.0]]), 1.0, 2.0)
    assert np.allclose(F[0], [1.2944, 0.6472], atol=1e-4), F
    A, B = pair_inverse(F, 1.0, 2.0, (-1, +1))
    assert np.allclose(A[0], [1, 0], atol=1e-9) and np.allclose(B[0], [0, 2], atol=1e-9)
    print("self-test OK: Fig. 35 example gives F = (%.4f, %.4f) and is inverted exactly" % tuple(F[0]))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", default="fisher_iris.csv")
    ap.add_argument("--class-col", default="class")
    ap.add_argument("--out", default="pcoc_vs_coc.png")
    ap.add_argument("--span", type=float, default=TAU, help="angular span for value range [0,1], radians")
    ap.add_argument("--no-collocate", action="store_true", help="give every attribute its own circle in PCoC")
    ap.add_argument("--highlight", type=int, default=None, help="row index of a case to highlight")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    selftest()

    df = pd.read_csv(args.csv)
    y = df[args.class_col].astype("category")
    feats = df.select_dtypes("number").drop(columns=["id"], errors="ignore")
    names, Xn = list(feats.columns), minmax(feats.values)
    y_codes, classes = y.cat.codes.values, list(y.cat.categories)

    collocate = not args.no_collocate
    err, frac = pcoc_roundtrip(Xn, attribute_radii(Xn.shape[1], collocate), args.span)
    print(f"round trip with stored side flags: max |error| = {err:.2e}")
    print(f"round trip with fixed (-1,+1) convention: {frac:.1%} of cases recovered")

    plot_comparison(Xn, y_codes, classes, names, collocate, args.span, args.highlight, args.out, args.show)
    plot_construction(Xn[args.highlight or 0], names, args.span,
                      args.out.replace(".png", "_construction.png"))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
