"""Figures for the Environmental Systems Research revision.

Reads the tables written by ``esr_revision_analyses.py`` and renders
  Fig10_added_value       -> manuscript Fig. 8
  Fig11_robustness_ext    -> Additional file 1: Fig. S3
  Fig12_profiles_years    -> Additional file 1: Fig. S4
with the same publication style and independently rendered subfigures as the
main pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SACC_FAST_REPLOT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import generate_sacc_paper_figures as G  # noqa: E402

T = G.SYSTEM_ROOT / "tables" / "esr_revision"
COLORS, EDGE, INK, MUTED = G.COLORS, G.EDGE, G.INK, G.MUTED
style_axis, numeric_ticks = G.style_axis, G.numeric_ticks


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(T / name, **kw)


def wrap(text: str, width: int = 18) -> str:
    import textwrap
    return textwrap.fill(text, width=width, break_long_words=False)


# ---------------------------------------------------------------------------
# Fig. 8: what the perturbation diagnostics add
# ---------------------------------------------------------------------------

def fig_added_value() -> list:
    lin = read("S05_lineage_definition.csv")
    leads = read("S16_label_flip_review_leads.csv", dtype={"Sector": str})
    fam = read("S16_family_reassigned_sectors.csv", dtype={"Sector": str})
    syn = read("S15_synthetic_summary.csv")
    rev = read("S14_release_revision_association.csv")

    def p1(ax):
        labels = ["Recorded\nentries", "Time-consistent\nseries", "GHGI\ntable", "Attribution\nfamily"]
        x = np.arange(len(lin))
        bars = ax.bar(x, lin["top50_emissions_HHI_overlap_with_primary_joint"], color=COLORS[1], edgecolor=EDGE,
                      width=0.62)
        for xi, b, med, n in zip(x, bars, lin["median_HHI"], lin["units"]):
            ax.text(xi, b.get_height() + 1.2, f"n={n}\nmedian HHI {med:.2f}", ha="center", fontsize=8.6, color=INK)
        ax.set_xticks(x, labels)
        style_axis(ax, ylabel="Shared top-50 accounts")
        ax.set_ylim(0, 66)

    def p2(ax):
        d = leads.copy()
        sc = ax.scatter(d["dominant_share"], d["max_flipping_share"].clip(lower=1e-4),
                        s=20 + 60 * np.sqrt(d["AnnualMean_MtCO2e"] / d["AnnualMean_MtCO2e"].max()),
                        c=[COLORS[int(a) - 1] for a in d["Archetype"]], edgecolors=EDGE, linewidths=0.4, alpha=0.85)
        ax.set_yscale("log")
        ax.axhline(0.05, color=EDGE, ls="--", lw=1.1)
        ax.text(0.02, 0.040, "5% of sector CO$_2$e", transform=ax.get_yaxis_transform(), fontsize=8.6,
                color=MUTED, va="top")
        positions = {"F01000": (0.04, 0.93), "483": (0.12, 0.77), "325412": (0.08, 0.30)}
        for code, xy in positions.items():
            r = d[d["Sector"] == code]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.annotate(wrap(r["SectorName"], 20), (r["dominant_share"], max(r["max_flipping_share"], 1e-4)),
                        xytext=xy, textcoords="axes fraction", fontsize=8.5,
                        arrowprops={"arrowstyle": "-", "color": EDGE, "lw": 0.7})
        style_axis(ax, xlabel="Share of dominant lineage", ylabel="Largest share of a\nlabel-changing lineage")
        ax.set_xlim(0, 1.02)

    def p3(ax):
        f = fam.drop_duplicates(["family", "Sector"]).copy()
        f["label"] = f["family"].map({"BEA_Detail_Use_SUT": "BEA use", "EIA_MECS_Energy": "EIA MECS", "Direct": "Direct"})
        for j, (nm, g) in enumerate(f.groupby("label")):
            ax.scatter(g["emission_rank"], g["HHI"], s=34, marker="os^"[j % 3],
                       facecolors=np.where(g["in_emissions_HHI_top50"], COLORS[j], "white"),
                       edgecolors=COLORS[j] if j else EDGE, linewidths=1.1, label=nm)
        ax.set_xscale("log")
        ax.axvline(50, color=EDGE, ls=":", lw=1.0)
        style_axis(ax, xlabel="Emissions rank", ylabel="Lineage HHI")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=9,
                  title="Open symbols: outside the emissions–HHI top 50", title_fontsize=8.5)
        ax.set_ylim(0, 1.05)

    def p4(ax):
        s = syn[syn["subset"] == "all affected"].set_index("kind")
        kinds = [k for k in ["key_permutation", "crosswalk_leakage", "scale_error"] if k in s.index]
        cols = ["recall_emissions", "recall_HHI", "recall_emissions_HHI", "recall_emissions_fragility"]
        names = ["Emissions", "HHI", "Emissions-HHI", "Emissions-fragility"]
        w = 0.19
        x = np.arange(len(kinds))
        for j, (c, nm) in enumerate(zip(cols, names)):
            ax.bar(x + (j - 1.5) * w, s.loc[kinds, c], w, color=COLORS[j], edgecolor=EDGE, label=nm)
        ax.set_xticks(x, ["Key\npermutation", "Crosswalk\nleakage", "Scale\nerror"][: len(kinds)])
        style_axis(ax, ylabel="Affected sectors in top 50")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9)

    def p5(ax):
        pooled = []
        sec = read("S15_synthetic_affected_sectors.csv")
        sec = sec[sec.erroneous_share > 0]  # sectors that receive mass from the erroneous lineage
        for nm, h in [("Profile-\naffected", sec[sec.profile_affected]),
                      ("Magnitude\nonly", sec[sec.magnitude_affected & ~sec.profile_affected]),
                      ("Erroneous lineage\n<5% of sector", sec[sec.erroneous_share < 0.05])]:
            pooled.append((nm, (h.share_rank == 1).mean(), (h.response_rank == 1).mean(), len(h)))
        x = np.arange(len(pooled))
        ax.bar(x - 0.18, [p[1] for p in pooled], 0.36, color=COLORS[2], edgecolor=EDGE, label="Largest emission share")
        ax.bar(x + 0.18, [p[2] for p in pooled], 0.36, color=COLORS[0], edgecolor=EDGE, label="Largest perturbation response")
        for xi, p in zip(x, pooled):
            ax.text(xi, max(p[1], p[2]) + 0.03, f"n={p[3]:,}", ha="center", fontsize=8.6, color=INK)
        ax.set_xticks(x, [p[0] for p in pooled])
        style_axis(ax, ylabel="Erroneous lineage\nranked first (fraction)")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=9)

    def p6(ax):
        order = ["Emissions", "HHI", "Emissions-HHI priority", "Emissions-fragility priority",
                 "Maximum lineage Aitchison shift"]
        labels = ["Emissions", "HHI", "Emissions–HHI", "Emissions–fragility", "Max. lineage\nAitchison shift"]
        y = np.arange(len(order))
        for j, (target, name, mk) in enumerate([("allocation_revision", "Allocation revision", "o"),
                                                ("profile_revision_grouped_F", "Profile revision", "s")]):
            r = rev[rev["target"] == target].set_index("diagnostic").reindex(order)
            yy = y + (j - 0.5) * 0.28
            ax.errorbar(r["spearman"], yy, xerr=[r["spearman"] - r["ci_low"], r["ci_high"] - r["spearman"]],
                        fmt=mk, color=COLORS[4 if j else 1], ecolor=EDGE, capsize=2.5, ms=6.5, mec=EDGE, label=name)
        ax.axvline(0, color=EDGE, ls="--", lw=1.0)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        style_axis(ax, xlabel="Spearman correlation with\nrelease-to-release revision")
        plt.setp(ax.get_yticklabels(), fontsize=8.8)
        ax.legend(loc="lower left", fontsize=8.5)

    return [p1, p2, p3, p4, p5, p6]


# ---------------------------------------------------------------------------
# Fig. S3: extended robustness
# ---------------------------------------------------------------------------

def fig_robustness() -> list:
    z = read("S04_zero_sensitivity.csv")
    k = read("S08_k_selection_2to8.csv")
    seeds = read("S08_seed_sacc.csv")
    opt = read("S08_seed_unperturbed_kmeans.csv")
    tests = read("S07_lineage_tests.csv")
    sev = read("S12_family_severity.csv")
    wts = read("S13_weights_and_sizes.csv")

    def p1(ax):
        short = {
            "Main rule: 0.65 x row minimum, cap 0.50": "Main",
            "Factor 0.10": "δ 0.10", "Factor 0.35": "δ 0.35", "Factor 0.90": "δ 0.90",
            "Cap 0.05": "cap 0.05", "Cap 0.10": "cap 0.10", "Cap 0.25": "cap 0.25", "Cap 0.90": "cap 0.90",
            "Column detection limit (0.65 x gas minimum)": "Gas DL", "Fixed share 1e-6": "1e-6",
            "Fixed share 1e-4": "1e-4", "Four-part amalgamation (CO2, CH4, N2O, F-gases)": "4-part",
        }
        y = np.arange(len(z))
        ax.barh(y, z["changed_vs_main"], color=COLORS[1], edgecolor=EDGE)
        for yi, (c, a) in enumerate(zip(z["changed_vs_main"], z["ARI_vs_main"])):
            ax.text(c + 2, yi, f"ARI {a:.2f}", va="center", fontsize=8.4, color=INK)
        ax.set_yticks(y, [short.get(v, v) for v in z["variant"]])
        ax.invert_yaxis()
        style_axis(ax, xlabel="Sectors reassigned vs main rule")
        ax.set_xlim(0, max(10, z["changed_vs_main"].max() * 1.45))
        plt.setp(ax.get_yticklabels(), fontsize=8.8)

    def p2(ax):
        ax.plot(k["K"], k["silhouette"], marker="o", lw=2.4, color=COLORS[1], label="Silhouette")
        ax.plot(k["K"], 1 - k["PAC"], marker="s", lw=2.4, color=COLORS[4], label="1 − PAC")
        ax.plot(k["K"], k["selection_score"], marker="D", lw=2.0, color=COLORS[0], label="Q(K)")
        ax.axvline(3, color=EDGE, ls="--", lw=1.3)
        ax.set_xticks(k["K"])
        style_axis(ax, xlabel="Number of clusters (K)", ylabel="Metric")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=9)

    def p3(ax):
        data = [seeds["ARI_vs_main"], opt[opt.n_init == 10]["ARI_vs_main"], opt[opt.n_init == 1]["ARI_vs_main"]]
        parts = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True)
        for b, c in zip(parts["boxes"], [COLORS[0], COLORS[2], COLORS[3]]):
            b.set_facecolor(c); b.set_edgecolor(EDGE)
        for key in ["whiskers", "caps", "medians"]:
            for it in parts[key]:
                it.set_color(EDGE)
        ax.set_xticks([1, 2, 3], ["SACC\n20 seeds", "K-means\nn_init=10", "K-means\nn_init=1"])
        style_axis(ax, ylabel="ARI with main partition")

    def p4(ax):
        order = ["<0.01%", "0.01-0.1%", "0.1-1%", ">=1%"]
        data = [tests[tests.mass_class == o]["ARI"].to_numpy() for o in order]
        for j, v in enumerate(data):
            xs = j + 1 + np.linspace(-0.15, 0.15, len(v)) if len(v) > 1 else [j + 1]
            ax.scatter(xs, v, s=22, color=COLORS[j], edgecolors=EDGE, linewidths=0.4)
            ax.text(j + 1, 1.012, f"n={len(v)}", ha="center", fontsize=8.6)
        ax.set_xticks(range(1, 5), ["<0.01%", "0.01–\n0.1%", "0.1–1%", "≥1%"])
        style_axis(ax, xlabel="Lineage share of national CO$_2$e", ylabel="Lineage-test ARI")
        ax.set_ylim(min(0.75, tests.ARI.min() - 0.02), 1.03)

    def p5(ax):
        for j, (fam, lab) in enumerate([("BEA_Detail_Use_SUT", "BEA use"), ("EIA_MECS_Energy", "EIA MECS"),
                                       ("Direct", "Direct")]):
            g = sev[(sev.family == fam) & (sev.feasible)].sort_values("retention")
            ax.plot(100 * g.retention, g.changed, marker="os^"[j], lw=2.2, color=COLORS[j], label=lab)
        style_axis(ax, xlabel="Retained share of family (%)", ylabel="Sectors reassigned")
        ax.legend(loc="upper right", fontsize=9)

    def p6(ax):
        w = wts[wts.weight_emissions != "arithmetic 0.5"].copy()
        w["weight_emissions"] = w["weight_emissions"].astype(float)
        for j, n in enumerate([25, 50, 100]):
            g = w[w.shortlist == n].sort_values("weight_emissions")
            ax.plot(g.weight_emissions, g.emissions_share_pct, marker="o", lw=2.2, color=COLORS[j], label=f"Top {n}")
        ax.axvline(0.5, color=EDGE, ls="--", lw=1.1)
        style_axis(ax, xlabel="Emissions weight w", ylabel="National CO$_2$e covered (%)")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, 100)

    return [p1, p2, p3, p4, p5, p6]


# ---------------------------------------------------------------------------
# Fig. S4: profile heterogeneity, continuous shifts and annual profiles
# ---------------------------------------------------------------------------

def fig_profiles() -> list:
    eta = read("S09_clr_coordinate_eta2.csv")
    dom = read("S09_dominant_gas_by_archetype.csv", index_col=0)
    coord = read("S09_truck_landfill_coordinates.csv")
    cont = read("S10_sector_continuous_diagnostics.csv", dtype={"Sector": str})
    years = read("S11_single_year_partitions.csv")
    pce = read("S11_PCE_by_year.csv")
    abbrev = {"Carbon dioxide": "CO$_2$", "Methane": "CH$_4$", "Nitrous oxide": "N$_2$O",
              "Carbon tetrafluoride": "CF$_4$", "Hexafluoroethane": "C$_2$F$_6$", "Perfluoropropane": "C$_3$F$_8$",
              "Perfluorocyclobutane": "c-C$_4$F$_8$", "Sulfur hexafluoride": "SF$_6$", "Nitrogen trifluoride": "NF$_3$",
              "HFCs and PFCs, unspecified": "HFC/PFC*"}

    def p1(ax):
        e = eta.sort_values("eta2")
        ax.barh(np.arange(len(e)), e["eta2"], color=COLORS[1], edgecolor=EDGE)
        ax.set_yticks(np.arange(len(e)), [abbrev.get(g, g) for g in e["gas"]])
        style_axis(ax, xlabel="Between-archetype share of CLR variance (η²)")
        plt.setp(ax.get_yticklabels(), fontsize=8.6)
        ax.set_xlim(0, 1)

    def p2(ax):
        d = dom.div(dom.sum(axis=1), axis=0)
        bottom = np.zeros(len(d))
        for j, c in enumerate(d.columns):
            lab = {"CO2": "CO$_2$", "CH4": "CH$_4$", "N2O": "N$_2$O"}.get(c, c)
            ax.bar(np.arange(len(d)), d[c], bottom=bottom, color=COLORS[j], edgecolor="white", width=0.6, label=lab)
            bottom += d[c].to_numpy()
        ax.set_xticks(np.arange(len(d)), [f"A{i}" for i in d.index])
        style_axis(ax, xlabel="Archetype", ylabel="Share of sectors")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, title="Dominant gas group",
                  title_fontsize=9)

    def p3(ax):
        x = np.arange(len(coord))
        ax.plot(x, coord["truck_clr"], marker="o", lw=1.8, color=COLORS[0], label="Truck transportation")
        ax.plot(x, coord["landfill_clr"], marker="s", lw=1.8, color=COLORS[2], label="Solid-waste landfill")
        ax.plot(x, coord["A1_centroid"], ls="--", lw=1.6, color=EDGE, label="A1 centroid")
        ax.plot(x, coord["A3_centroid"], ls=":", lw=1.8, color=COLORS[4], label="A3 centroid")
        ax.set_xticks(x, [abbrev.get(g, g.replace("HFC-", "")) for g in coord["gas"]], rotation=70)
        style_axis(ax, ylabel="CLR coordinate")
        plt.setp(ax.get_xticklabels(), fontsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=8.5)

    def p4(ax):
        c = cont.copy()
        stable = c["Stability"] >= 1 - 1e-12
        ax.scatter(c.loc[~stable, "max_abs_emission_change"] * 100, c.loc[~stable, "max_Aitchison_shift"], s=26,
                   color=COLORS[0], edgecolors=EDGE, linewidths=0.4, label="Label changes")
        ax.scatter(c.loc[stable, "max_abs_emission_change"] * 100, c.loc[stable, "max_Aitchison_shift"], s=14,
                   color=COLORS[1], alpha=0.6, edgecolors="none", label="Stable label")
        ax.set_xscale("symlog", linthresh=1)
        ax.axvline(10, color=EDGE, ls="--", lw=1.0)
        style_axis(ax, xlabel="Largest emission change (%)", ylabel="Largest Aitchison shift")
        ax.legend(loc="upper left", fontsize=9)

    def p5(ax):
        ax.bar(years["year"], years["SACC_changed"], color=COLORS[3], edgecolor=EDGE, width=0.6)
        for x, v, a in zip(years["year"], years["SACC_changed"], years["SACC_ARI_vs_reference"]):
            ax.text(x, v + 0.4, f"ARI {a:.3f}", ha="center", fontsize=8.4)
        ax.set_xticks(years["year"])
        style_axis(ax, xlabel="Single inventory year", ylabel="Sectors differing from\nfive-year partition")
        ax.set_ylim(0, max(5, years["SACC_changed"].max() * 1.35))

    def p6(ax):
        ax.plot(pce["year"], pce["dist_A1"], marker="o", lw=2.2, color=COLORS[0], label="Distance to A1")
        ax.plot(pce["year"], pce["dist_A3"], marker="s", lw=2.2, color=COLORS[2], label="Distance to A3")
        ax2 = ax.twinx()
        ax2.bar(pce["year"], pce["Fgas_pct"], color=COLORS[1], alpha=0.35, width=0.5)
        ax2.set_ylabel("F-gas share (%)", color=MUTED)
        ax.set_xticks(pce["year"])
        style_axis(ax, xlabel="Inventory year", ylabel="PCE Aitchison distance")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    return [p1, p2, p3, p4, p5, p6]


def main() -> int:
    for number, stem, builder in [(10, "added_value", fig_added_value), (11, "robustness_extensions", fig_robustness),
                                  (12, "profiles_and_years", fig_profiles)]:
        G.export_ordered_figure(number, stem, builder(), None)
        print(f"Exported Fig{number}_{stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
