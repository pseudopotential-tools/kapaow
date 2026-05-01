"""Periodic table visualization of PAO optimization results."""

import json
import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib.cm as cm
import pandas as pd
from ase.data import atomic_numbers
from bokeh.io import export_png, export_svg, save
from bokeh.transform import dodge
from periodic_trends import plotter
from tqdm import tqdm
from upf_tools import UPFDict

from kapaow.basis import AtomicBasis
from kapaow.extend import BasisExtension, BasisExtensionViaAddition, parse_extension
from kapaow.plotting import COLORMAP, REVTEX_DOUBLE_COLUMN_WIDTH

logger = logging.getLogger(__name__)

_BOKEH_CMAP = cm.get_cmap(COLORMAP)

_BOKEH_WIDTH_PX = 1050
# Scale factor so exported PNG has the correct physical width for RevTeX double column at 300 DPI
_BOKEH_SCALE_FACTOR = REVTEX_DOUBLE_COLUMN_WIDTH * 300 / _BOKEH_WIDTH_PX


# Noble gas cores by atomic number
_NOBLE_GAS_CORES = [
    (86, "Rn"),
    (54, "Xe"),
    (36, "Kr"),
    (18, "Ar"),
    (10, "Ne"),
    (2, "He"),
]


def _basis_annotation(
    upf_path: Path,
    extension: BasisExtension | None = None,
) -> str:
    """Build an annotation like '[He]2s2p+3s' for a given UPF file.

    If *extension* is None no extended orbitals are appended and the
    annotation is just the core-suppressed original basis.
    """
    upf_dict = UPFDict.from_upf(upf_path)
    basis = AtomicBasis.from_upf(upf_path)
    if extension is not None:
        extended = extension.extend_atomic(basis)
        added = [s for s in extended.subshells if s not in basis.subshells]
    else:
        added = []

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

    if not added:
        return orig_str

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


def _collect_projectability_data(
    json_directory: Path,
) -> list:
    """Scan bayes_opt optimizer logs and return ``[element, max target]`` rows.

    Each ``<element>.log.json`` file is loaded by the same
    :class:`BayesianOptimization` instance type used during optimisation,
    and the best (max) target is extracted. Stdout / stderr from
    ``bayes_opt`` is suppressed so a directory of logs doesn't flood the
    terminal.

    The bayes_opt machinery lives in :mod:`kapaow._experimental.optimize`,
    so this helper is only callable when the ``[experimental]`` extras
    are installed.
    """
    from kapaow._experimental.optimize import create_optimizer

    rows: list = []
    for json_file in tqdm(list(json_directory.glob("*.log.json"))):
        element = json_file.stem[:-4]
        optimizer = create_optimizer()
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                optimizer.load_state(json_file)
        rows.append([element, optimizer.max["target"]])
    return rows


def plot_periodic_table(
    json_directory: Path,
    output: Path | None = None,
) -> None:
    """Plot a periodic table colored by best projectability per element.

    Reads bayes_opt optimizer logs (``<element>.log.json``) from
    *json_directory* and renders a Bokeh periodic table whose colour
    encodes the optimum projectability score. Annotations are not
    populated because the optimizer logs don't carry the UPF path used
    for the run.
    """
    plot_rows = _collect_projectability_data(json_directory)
    _render_periodic_table(
        plot_rows,
        annotations={},
        cbar_title="Best projectability",
        output=output,
    )


def _collect_pareto_data(
    pareto_directory: Path,
    threshold_ha: float,
    threshold_ry: float,
) -> tuple[list, list, dict[str, str], set[str]]:
    """Scan Pareto JSON files and return data for plotting."""
    plot_rows: list = []
    table_rows: list = []
    annotations: dict[str, str] = {}
    unmodified_elements: set[str] = set()
    for json_file in sorted(pareto_directory.glob("*.json")):
        element = json_file.stem
        with open(json_file) as f:
            raw = json.load(f)

        if "upf_path" in raw:
            upf_path = Path(raw["upf_path"])
            if upf_path.exists():
                annotations[element] = _basis_annotation(
                    upf_path, extension=BasisExtensionViaAddition()
                )

        pareto_data = [p for p in raw["points"] if p["pareto"]]
        candidates = [p for p in pareto_data if p["max_energy_shift"] < threshold_ha]
        if candidates:
            best = min(candidates, key=lambda p: p["spread"])
            rc = best["rc"]
            ri_factor = best["ri_factor"]
            r_half = (ri_factor + 1) / 2 * rc
            width = rc * (1 - ri_factor) / 2
            spread = best["spread"]
            plot_rows.append([element, spread])
            table_rows.append([element, r_half, width, spread])

            if all(not p["modified_by_confinement"] for p in candidates):
                unmodified_elements.add(element)
        else:
            logger.info(
                "  %s: no point below %s Ry threshold",
                element,
                threshold_ry,
            )
    return plot_rows, table_rows, annotations, unmodified_elements


def _add_annotations(p, annotations: dict[str, str]) -> None:
    """Add basis annotations below the element symbol on the Bokeh plot."""
    source = p.renderers[0].data_source
    anno_text = [annotations.get(sym, "") for sym in source.data["sym"]]
    source.data["annotation"] = anno_text
    p.text(
        x=dodge("group", -0.4, range=p.x_range),
        y=dodge("period", -0.3, range=p.y_range),
        text="annotation",
        source=source,
        text_font_size="5pt",
        text_align="left",
        text_baseline="middle",
        color="black",
    )


def _add_unmodified_boxes(p, unmodified_elements: set[str]) -> None:
    """Draw boxes around elements unmodified by confinement."""
    from bokeh.models import ColumnDataSource

    source = p.renderers[0].data_source
    box_groups = [
        source.data["group"][i]
        for i, sym in enumerate(source.data["sym"])
        if sym in unmodified_elements
    ]
    box_periods = [
        source.data["period"][i]
        for i, sym in enumerate(source.data["sym"])
        if sym in unmodified_elements
    ]
    box_source = ColumnDataSource(
        data={"group": box_groups, "period": box_periods},
    )
    p.rect(
        "group",
        "period",
        0.9,
        0.9,
        source=box_source,
        fill_alpha=0,
        line_color="black",
        line_width=2,
    )


def _dump_latex_table(table_rows: list, output: Path) -> None:
    """Write a LaTeX longtable with element, r_half, width, and spread."""
    table_df = pd.DataFrame(
        table_rows,
        columns=[
            "Element",
            r"$r_{1/2}$ (Bohr)",
            r"Width (Bohr)",
            r"Spread (Bohr$^2$)",
        ],
    )
    table_df["Z"] = table_df["Element"].map(atomic_numbers)
    table_df = table_df.sort_values("Z").drop(columns="Z").reset_index(drop=True)
    tex_path = output.with_suffix(".tex")
    # Build longtable with siunitx d-type columns for floats
    n_float_cols = len(table_df.columns) - 1
    col_spec = "l" + " ".join(
        ["D{.}{.}{3.3}"] * n_float_cols,
    )
    header = " & ".join(
        [table_df.columns[0]] + [rf"\multicolumn{{1}}{{c}}{{{c}}}" for c in table_df.columns[1:]]
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
    logger.info("LaTeX table saved to %s", tex_path)


def _render_periodic_table(
    plot_rows: list,
    annotations: dict[str, str],
    cbar_title: str,
    output: Path | None,
    unmodified_elements: set[str] | None = None,
) -> None:
    """Build and render a Bokeh periodic-table plot from pre-collected rows."""
    if output is not None:
        if output.suffix not in {".png", ".svg", ".pdf", ".html"}:
            raise ValueError(f"Unsupported output format: {output.suffix}")

    plot_df = pd.DataFrame(plot_rows, columns=["Element", "Score"])
    p = plotter(
        plot_df,
        "Element",
        "Score",
        show=False,
        extended=False,
        periods_remove=[7],
        width=_BOKEH_WIDTH_PX,
        cbar_fontsize=8,
        cmap=_BOKEH_CMAP,
        cbar_title=cbar_title,
        rescale_canvas=True,
    )

    _add_annotations(p, annotations)

    if unmodified_elements:
        _add_unmodified_boxes(p, unmodified_elements)

    _scale_fonts(p)

    if output is None:
        from bokeh.io import show as show_

        show_(p)
    _save_or_show(p, output)


def plot_pareto_periodic_table(
    pareto_directory: Path,
    output: Path | None = None,
    threshold_ry: float = 0.02,
) -> None:
    """Plot a periodic table colored by smallest spread on the Pareto front.

    Select points for which the max energy shift is below *threshold_ry*
    (in Rydberg). Also dumps a .tex file (same stem as *output*) with a table of
    element, r_half, width, and spread.
    """
    threshold_ha = threshold_ry / 2  # Convert Ry to Ha

    plot_rows, table_rows, annotations, unmodified_elements = _collect_pareto_data(
        pareto_directory,
        threshold_ha,
        threshold_ry,
    )

    _render_periodic_table(
        plot_rows,
        annotations,
        cbar_title="Spread of added PAO (Bohr^2)",
        output=output,
        unmodified_elements=unmodified_elements,
    )

    if output is not None:
        _dump_latex_table(table_rows, output)


def _collect_rc_data(
    rc_directory: Path,
) -> tuple[list, dict[str, str]]:
    """Scan rc-search JSON files and return plot rows and basis annotations."""
    plot_rows: list = []
    annotations: dict[str, str] = {}
    for json_file in sorted(rc_directory.glob("*.json")):
        element = json_file.stem
        with open(json_file) as f:
            raw = json.load(f)

        if "upf_path" in raw:
            upf_path = Path(raw["upf_path"])
            if upf_path.exists():
                add_spec = tuple(raw.get("add", ()) or ())
                extension = parse_extension(add_spec) if add_spec else None
                annotations[element] = _basis_annotation(upf_path, extension=extension)

        rc_value = raw.get("rc")
        if rc_value is None:
            logger.info("  %s: no rc value in %s", element, json_file)
            continue
        plot_rows.append([element, rc_value])
    return plot_rows, annotations


def plot_rc_periodic_table(
    rc_directory: Path,
    output: Path | None = None,
) -> None:
    """Plot a periodic table colored by the smallest rc found by the rc search.

    Reads JSON files produced by ``kapaow optimize rc`` from
    *rc_directory* (one per element, named ``<element>.json``).
    """
    plot_rows, annotations = _collect_rc_data(rc_directory)
    _render_periodic_table(
        plot_rows,
        annotations,
        cbar_title="minimal cutoff radius (Bohr)",
        output=output,
    )
