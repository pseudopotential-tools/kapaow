import json
import os
from contextlib import redirect_stdout, redirect_stderr
from typing import Callable

import pandas as pd
from bokeh.io import export_png, export_svg, save
from bokeh.transform import dodge
from periodic_trends import plotter
from tqdm import tqdm
from ase_koopmans.data import atomic_numbers
from upf_tools import UPFDict

from pao_plusplus.basis import AtomicBasis, Subshell, ordered_subshells
from pao_plusplus.extend import BasisExtensionViaAddition
from pao_plusplus.optimize import create_optimizer
import matplotlib.cm as cm

from pao_plusplus.plotting import COLORMAP, REVTEX_DOUBLE_COLUMN_WIDTH

_BOKEH_CMAP = cm.get_cmap(COLORMAP)

from pathlib import Path

_BOKEH_WIDTH_PX = 1050
# Scale factor so exported PNG has the correct physical width for RevTeX double column at 300 DPI
_BOKEH_SCALE_FACTOR = REVTEX_DOUBLE_COLUMN_WIDTH * 300 / _BOKEH_WIDTH_PX


# Noble gas cores by atomic number
_NOBLE_GAS_CORES = [
    (86, "Rn"), (54, "Xe"), (36, "Kr"), (18, "Ar"), (10, "Ne"), (2, "He"),
]


def _basis_annotation(upf_path: Path) -> str:
    """Build an annotation like '[He]2s2p + 3s' for a given UPF file."""
    upf_dict = UPFDict.from_upf(upf_path)
    basis = AtomicBasis.from_upf(upf_path)
    ext = BasisExtensionViaAddition()
    extended = ext.extend_atomic(basis)
    added = [s for s in extended.subshells if s not in basis.subshells]

    element = upf_dict["header"]["element"].strip()
    z = int(upf_dict["header"]["z_valence"])
    z_total = atomic_numbers[element]
    z_core = z_total - z

    # Find noble gas core
    core_label = ""
    for z_ng, sym in _NOBLE_GAS_CORES:
        if z_core >= z_ng:
            core_label = f"[{sym}]"
            break

    l_names = {0: "s", 1: "p", 2: "d", 3: "f"}

    # Original basis subshells
    orig_parts = []
    for s in basis.subshells:
        orig_parts.append(f"{s.n}{l_names[s.l.value]}")
    orig_str = core_label + "".join(orig_parts)

    # Added subshells
    added_parts = []
    for s in added:
        added_parts.append(f"{s.n}{l_names[s.l.value]}")
    added_str = ",".join(added_parts)

    return f"{orig_str}+{added_str}"


def _scale_fonts(p, factor: float = 0.85) -> None:
    """Scale all font sizes in a Bokeh figure by *factor*."""
    from bokeh.models import Text
    for renderer in p.renderers:
        glyph = getattr(renderer, "glyph", None)
        if isinstance(glyph, Text) and glyph.text_font_size:
            size_str = glyph.text_font_size
            if size_str.endswith("pt"):
                new_size = float(size_str[:-2]) * factor
                glyph.text_font_size = f"{new_size:.1f}pt"
    if p.title and p.title.text_font_size:
        size_str = p.title.text_font_size
        if size_str.endswith("pt"):
            new_size = float(size_str[:-2]) * factor
            p.title.text_font_size = f"{new_size:.1f}pt"


def _save_or_show(p, output: Path | None) -> None:
    """Save a Bokeh plot to file or show it interactively."""
    if output is None:
        return
    if output.suffix == ".png":
        export_png(p, filename=str(output), scale_factor=_BOKEH_SCALE_FACTOR)
    elif output.suffix == ".svg":
        export_svg(p, filename=str(output))
    elif output.suffix == ".pdf":
        import tempfile
        import cairosvg
        with tempfile.NamedTemporaryFile(suffix=".svg") as tmp:
            export_svg(p, filename=tmp.name)
            svg_bytes = Path(tmp.name).read_bytes()
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(output))
    else:
        save(p, filename=str(output))


def plot_periodic_table(extract_data_from_optimizer: Callable, json_directory: Path,
                        output: Path | None = None) -> None:

    if output is not None:
        if output.suffix not in {".png", ".svg", ".pdf", ".html"}:
            raise ValueError(f"Unsupported output format: {output.suffix}")

    rows = []
    for json_file in tqdm(list(json_directory.glob('*.log.json'))):
        element = json_file.stem[:-4]
        optimizer = create_optimizer()
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                optimizer.load_state(json_file)
        rows.append([element, extract_data_from_optimizer(optimizer)])
    df = pd.DataFrame(rows, columns=["Element", "Score"])

    p = plotter(df, "Element", "Score", show=output is None, extended=False,
                periods_remove=[7], width=_BOKEH_WIDTH_PX, cbar_fontsize=8,
                cmap=_BOKEH_CMAP, rescale_canvas=True)
    _scale_fonts(p)
    _save_or_show(p, output)


def plot_pareto_periodic_table(
    pareto_directory: Path,
    output: Path | None = None,
    threshold_ry: float = 0.02,
) -> None:
    """Plot a periodic table colored by the smallest spread on the Pareto front
    for which the max energy shift is below *threshold_ry* (in Rydberg).

    Also dumps a .tex file (same stem as *output*) with a table of
    element, r_half, width, and spread.
    """

    if output is not None:
        if output.suffix not in {".png", ".svg", ".pdf", ".html"}:
            raise ValueError(f"Unsupported output format: {output.suffix}")

    threshold_ha = threshold_ry / 2  # Convert Ry to Ha

    plot_rows = []
    table_rows = []
    annotations = {}
    unmodified_elements: set[str] = set()
    for json_file in sorted(pareto_directory.glob("*.json")):
        element = json_file.stem
        with open(json_file) as f:
            raw = json.load(f)

        if "upf_path" in raw:
            upf_path = Path(raw["upf_path"])
            if upf_path.exists():
                annotations[element] = _basis_annotation(upf_path)

        pareto_data = [p for p in raw["points"] if p["pareto"]]
        candidates = [
            p for p in pareto_data if p["max_energy_shift"] < threshold_ha
        ]
        if candidates:
            best = min(candidates, key=lambda p: p["spread"])
            rc = best["rc"]
            ri_factor = best["ri_factor"]
            r_half = (ri_factor + 1) / 2 * rc
            width = rc * (1 - ri_factor) / 2
            spread = best["spread"]
            plot_rows.append([element, spread])
            table_rows.append([element, r_half, width, spread])

            # Check if all below-threshold points are unmodified by confinement
            if all(
                not p["modified_by_confinement"] for p in candidates
            ):
                unmodified_elements.add(element)
        else:
            print(f"  {element}: no point below {threshold_ry} Ry threshold")

    plot_df = pd.DataFrame(plot_rows, columns=["Element", "Score"])
    p = plotter(plot_df, "Element", "Score", show=False, extended=False,
                periods_remove=[7], width=_BOKEH_WIDTH_PX, cbar_fontsize=8,
                cmap=_BOKEH_CMAP, cbar_title="Spread of added PAO (Bohr²)",
                rescale_canvas=True)

    # Add basis annotations below the element symbol
    source = p.renderers[0].data_source
    anno_text = [annotations.get(sym, "") for sym in source.data["sym"]]
    source.data["annotation"] = anno_text
    p.text(
        x=dodge("group", -0.4, range=p.x_range),
        y=dodge("period", -0.3, range=p.y_range),
        text="annotation", source=source,
        text_font_size="5pt", text_align="left", text_baseline="middle",
        color="black",
    )

    # Draw boxes around elements unmodified by confinement
    if unmodified_elements:
        source = p.renderers[0].data_source
        box_groups = [
            source.data["group"][i] for i, sym in enumerate(source.data["sym"])
            if sym in unmodified_elements
        ]
        box_periods = [
            source.data["period"][i] for i, sym in enumerate(source.data["sym"])
            if sym in unmodified_elements
        ]
        from bokeh.models import ColumnDataSource
        box_source = ColumnDataSource(data={"group": box_groups, "period": box_periods})
        p.rect(
            "group", "period", 0.9, 0.9, source=box_source,
            fill_alpha=0, line_color="black", line_width=2,
        )

    _scale_fonts(p)

    if output is None:
        from bokeh.io import show as show_
        show_(p)
    _save_or_show(p, output)

    # Dump a LaTeX table
    if output is not None:
        table_df = pd.DataFrame(
            table_rows,
            columns=["Element", r"$r_{1/2}$ (Bohr)", r"Width (Bohr)",
                      r"Spread (Bohr$^2$)"],
        )
        table_df["Z"] = table_df["Element"].map(atomic_numbers)
        table_df = table_df.sort_values("Z").drop(columns="Z").reset_index(drop=True)
        tex_path = output.with_suffix(".tex")
        # Build longtable with siunitx d-type columns for floats
        n_float_cols = len(table_df.columns) - 1  # all columns except Element
        col_spec = "l" + " ".join(["D{.}{.}{3.3}"] * n_float_cols)
        header = " & ".join(
            [table_df.columns[0]]
            + [rf"\multicolumn{{1}}{{c}}{{{c}}}" for c in table_df.columns[1:]]
        )
        lines = [
            r"\begin{longtable}{" + col_spec + "}",
            r"\hline",
            header + r" \\",
            r"\hline",
            r"\endfirsthead",
            r"\hline",
            header + r" \\",
            r"\hline",
            r"\endhead",
            r"\hline",
            r"\endfoot",
        ]
        for _, row in table_df.iterrows():
            vals = [str(row.iloc[0])] + [f"{v:.2f}" for v in row.iloc[1:]]
            lines.append(" & ".join(vals) + r" \\")
        lines.append(r"\end{longtable}")
        tex_path.write_text("\n".join(lines) + "\n")
        print(f"LaTeX table saved to {tex_path}")


