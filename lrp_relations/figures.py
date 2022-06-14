"""Module with functions for plotting."""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union

# from IPython import get_ipython
import matplotlib as mpl
import matplotlib.figure as mpl_figure
import numpy as np
import seaborn as sns


def get_figure_size(
    fraction: float = 0.5,
    width: float = 469.75499,  # tmlr width
    ratio: float = (5**0.5 - 1) / 2,  # gold ratio
    subplots: tuple[int, int] = (1, 1),
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Args:
        width: float or string
                Document width in points, or string of predined document type
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy
        ratio: Ratio of plot
        subplots: array-like, optional
                The number of rows and columns of subplots.

    Returns:
        fig_dim: tuple Dimensions of figure in inches
    """

    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "pnas":
        width_pt = 246.09686
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def get_latex_defs(**dictionary: dict[str, str]) -> str:
    defs = ""
    for key, val in dictionary.items():
        if "_" in key:
            key = underscore_to_camelcase(key)
        assert key.isalpha(), f"not a valid latex macro name: {key}"
        defs += f"\\newcommand{{\\{key}}}{{{val}}}\n"
    return defs


def equal_vmin_vmax(x: np.ndarray) -> dict[str, float]:
    vmax = float(np.abs(x).max())
    return {"vmin": -vmax, "vmax": vmax}


def export_latex_defs(filename: str, **dictionary: dict[str, str]):
    with open(filename, "w") as f:
        f.write(get_latex_defs(**dictionary))


def underscore_to_camelcase(value: str) -> str:
    def camelcase():
        yield str
        while True:
            yield str.capitalize

    c = camelcase()
    return "".join(next(c)(x) if x else "_" for x in value.split("_"))


@contextlib.contextmanager
def latexify(
    dark_gray: str = ".15",
    light_gray: str = ".8",
    small_size: int = 8,
    tiny_size: int = 7,
    linewidth_thin: float = 0.33,
    linewidth: float = 0.5,
    n_colors: Optional[int] = None,
):
    style = latex_style(
        dark_gray=dark_gray,
        light_gray=light_gray,
        small_size=small_size,
        tiny_size=tiny_size,
        linewidth_thin=linewidth_thin,
        linewidth=linewidth,
    )

    with mpl_style(style), sns.color_palette(
        "colorblind", n_colors=n_colors  # type: ignore
    ):
        yield


@contextlib.contextmanager
def mpl_style(style: dict[str, Any]):
    mpl_orig = copy.deepcopy(mpl.rcParams)
    for key, value in style.items():
        mpl.rcParams[key] = value
    try:
        yield
    finally:
        for key, value in mpl_orig.items():
            mpl.rcParams[key] = value


def latex_style(
    dark_gray: str = ".15",
    light_gray: str = ".8",
    small_size: int = 8,
    tiny_size: int = 7,
    linewidth_thin: float = 0.33,
    linewidth: float = 0.5,
) -> dict[str, Any]:
    # Common parameters
    return {
        "figure.facecolor": "white",
        "axes.labelcolor": dark_gray,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": dark_gray,
        "ytick.color": dark_gray,
        "axes.axisbelow": True,
        "grid.linestyle": "-",
        "text.color": dark_gray,
        # "font.family": ["serif"],
        # "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans",
        #                     "Bitstream Vera Sans", "sans-serif"],
        "font.family": "serif",
        # "font.serif": ['Times', "DejaVu Sans"],
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "image.cmap": "rocket",
        "xtick.top": False,
        "ytick.right": False,
        "axes.facecolor": "white",
        "xtick.major.width": linewidth,
        "xtick.minor.width": linewidth,
        "ytick.major.width": linewidth,
        "ytick.minor.width": linewidth,
        "grid.linewidth": linewidth_thin,
        "axes.linewidth": linewidth_thin,
        "lines.linewidth": linewidth,
        "lines.markersize": linewidth_thin,
        "patch.linewidth": linewidth_thin,
        "xtick.bottom": True,
        "ytick.left": True,
        "axes.facecolor": "white",
        "axes.edgecolor": dark_gray,
        "grid.color": light_gray,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.labelsize": small_size,
        "font.size": small_size,
        "axes.titlesize": small_size,
        "legend.fontsize": small_size,
        "xtick.labelsize": tiny_size,
        "ytick.labelsize": tiny_size,
    }


def display_pdf(fname: str, iframe_size: tuple[int, int] = (800, 400)):
    from IPython.display import IFrame, display

    display(IFrame(fname, *iframe_size))


def savefig_pdf(
    fname: str,
    figure: mpl_figure.Figure,
    display: bool = False,
    iframe_size: tuple[int, int] = (800, 400),
    **kwargs: Any,
):
    figure.savefig(fname, bbox_inches="tight", pad_inches=0.01, **kwargs)
    if display:
        display_pdf(fname, iframe_size)


def savefig_pgf(
    figure: mpl_figure.Figure,
    fname: Union[str, Path],
    display: bool = False,
    iframe_size: tuple[int, int] = (800, 400),
    pdf: bool = True,
    **kwargs: Any,
):
    figure.savefig(fname, bbox_inches="tight", pad_inches=0.03, **kwargs)
    if pdf or display:
        pdf_fname, _ = os.path.splitext(fname)
        pdf_fname += ".pdf"
        savefig_pdf(pdf_fname, figure, **kwargs)
    if display:
        display_pdf(pdf_fname)


@contextlib.contextmanager
def overwrite_dir(*path: str, upload: bool = False):
    fullpath = os.path.join(*path)
    if os.path.exists(fullpath):
        shutil.rmtree(fullpath)
    os.makedirs(fullpath)
    yield fullpath
    if upload:
        raise NotImplementedError()
