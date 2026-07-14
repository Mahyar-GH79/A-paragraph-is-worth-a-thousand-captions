"""Plot styling and configuration metadata shared by the paper's figures and tables.

Importing this module selects matplotlib's non-interactive ``Agg`` backend, so every
analysis module renders without a display.

The training configurations are referred to by the tags ``cfg1`` ... ``cfg10``
throughout the project; this module is the single source of truth for their ordering,
colours, markers and human-readable names.
"""


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (the backend must be selected first)

# --------------------------------------------------------------------------------------
# matplotlib style
# --------------------------------------------------------------------------------------

#: rcParams every figure shares. Per-figure deviations go through ``use_paper_style``.
BASE_RC: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def use_paper_style(**overrides: object) -> None:
    """Apply the shared rcParams, then any per-figure overrides."""
    plt.rcParams.update(BASE_RC)
    plt.rcParams.update(overrides)


# --------------------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------------------

#: The ten fine-tuning configurations, in the order they appear in every figure/table.
CFG_ORDER: list[str] = [f"cfg{i}" for i in range(1, 11)]

#: Configuration order for figures that also show the pretrained BLIP text tower.
CFG_ORDER_WITH_BASELINE: list[str] = ["baseline"] + CFG_ORDER

#: Row order for the retrieval tables: pretrained baselines first, then the configs.
TABLE_ROW_ORDER: list[str] = [
    "baseline_blip",
    "baseline_clip",
    "baseline_longclip_b",
    "baseline_longclip_l",
] + CFG_ORDER

#: Colour-blind-safe palette (Wong / Tol), keyed by config tag and baseline tag.
COLORS: dict[str, str] = {
    "baseline": "#7F7F7F",
    "clip": "#AAAAAA",
    "baseline_blip": "#888888",
    "baseline_clip": "#BBBBBB",
    "cfg1": "#0072B2",
    "cfg2": "#56B4E9",
    "cfg3": "#009E73",
    "cfg4": "#D55E00",
    "cfg5": "#E69F00",
    "cfg6": "#CC79A7",
    "cfg7": "#882255",
    "cfg8": "#44AA99",
    "cfg9": "#332288",
    "cfg10": "#117733",
}

MARKERS: dict[str, str] = {
    "baseline": "*",
    "clip": "X",
    "baseline_blip": "*",
    "baseline_clip": "*",
    "cfg1": "o",
    "cfg2": "s",
    "cfg3": "^",
    "cfg4": "D",
    "cfg5": "P",
    "cfg6": "v",
    "cfg7": "X",
    "cfg8": "h",
    "cfg9": "p",
    "cfg10": ">",
}

#: What each config trains on, spelled out (line plots, legends with room to spare).
CFG_DESCRIPTIONS: dict[str, str] = {
    "cfg1": "orig cap",
    "cfg2": "rand pos",
    "cfg3": "orig+rand pos",
    "cfg4": "paragraph",
    "cfg5": "orig+para",
    "cfg6": "rand pos+para",
    "cfg7": "orig+pos+neg",
    "cfg8": "orig+all5pos",
    "cfg9": "orig+5pos+para",
    "cfg10": "orig+rpos+para",
}

#: The same, abbreviated for dense axes (t-SNE grids, grouped bar charts).
CFG_DESCRIPTIONS_COMPACT: dict[str, str] = {
    "cfg1": "orig cap",
    "cfg2": "rand pos",
    "cfg3": "orig+rpos",
    "cfg4": "paragraph",
    "cfg5": "orig+para",
    "cfg6": "rpos+para",
    "cfg7": "pos+neg",
    "cfg8": "all5pos",
    "cfg9": "5pos+para",
    "cfg10": "o+rp+para",
}

#: Pretrained-model labels for figures that name them as "pretrained". Both the
#: ``baseline``/``clip`` tags (single-model runs) and the ``baseline_blip``/
#: ``baseline_clip`` tags (eval-result files) map to the same text.
PRETRAINED_LABELS: dict[str, str] = {
    "baseline": "BLIP$_0$ (pretrained)",
    "clip": "CLIP$_0$ (pretrained)",
    "baseline_blip": "BLIP$_0$ (pretrained)",
    "baseline_clip": "CLIP$_0$ (pretrained)",
}

#: Pretrained-model labels for dense figures.
PRETRAINED_LABELS_COMPACT: dict[str, str] = {"baseline": "BLIP$_0$"}

#: Table rows: (model name, training text) per config.
CFG_TABLE_DISPLAY: dict[str, tuple] = {
    "baseline_blip": ("BLIP$_0$ (pretrained)", "---"),
    "baseline_clip": ("CLIP$_0$ (pretrained)", "---"),
    "baseline_longclip_b": ("Long-CLIP-B", "---"),
    "baseline_longclip_l": ("Long-CLIP-L", "---"),
    "cfg1": ("CFG1", "Original caption"),
    "cfg2": ("CFG2", "Random positive caption"),
    "cfg3": ("CFG3", "Original + rand positive"),
    "cfg4": ("CFG4", "Paragraph only"),
    "cfg5": ("CFG5", "Original cap + paragraph"),
    "cfg6": ("CFG6", "Rand pos + paragraph"),
    "cfg7": ("CFG7", "Original + pos + hard neg"),
    "cfg8": ("CFG8", "Original + all 5 positives"),
    "cfg9": ("CFG9", "Orig + all 5 pos + paragraph"),
    "cfg10": ("CFG10", "Orig + rand pos + paragraph"),
}


def cfg_labels(compact: bool = False, prefix: str = "C") -> dict[str, str]:
    """Legend labels for the ten configs, e.g. ``{"cfg5": "C5: orig+para"}``.

    Args:
        compact: use the abbreviated descriptions.
        prefix: label prefix, ``"C"`` (default) or ``"CFG"``.
    """
    descriptions = CFG_DESCRIPTIONS_COMPACT if compact else CFG_DESCRIPTIONS
    return {tag: f"{prefix}{tag[3:]}: {text}" for tag, text in descriptions.items()}


# --------------------------------------------------------------------------------------
# Models and datasets
# --------------------------------------------------------------------------------------

#: The five models compared in the DOCCI qualitative and saliency figures.
MODEL_TAGS: list[str] = ["blip0", "clip", "longclip_b", "longclip_l", "cfg5"]

MODEL_LABELS: dict[str, str] = {
    "blip0": "BLIP$_0$",
    "clip": "CLIP$_0$",
    "longclip_b": "Long-CLIP-B",
    "longclip_l": "Long-CLIP-L",
    "cfg5": "CFG5 (ours)",
}

#: Benchmarks, in the order they are plotted. Short-caption first, then long-description.
DATASET_ORDER: list[str] = ["Flickr30k", "COCO", "ShareGPT4V", "DOCCI"]

#: Bar styling per dataset; the hatches keep the grouped bars legible in greyscale.
DATASET_STYLE: dict[str, dict[str, str]] = {
    "Flickr30k": {"color": "#4E79A7", "hatch": "", "edgecolor": "#3A5F8A"},
    "COCO": {"color": "#F28E2B", "hatch": "//", "edgecolor": "#C97520"},
    "ShareGPT4V": {"color": "#E15759", "hatch": "xx", "edgecolor": "#B84547"},
    "DOCCI": {"color": "#76B7B2", "hatch": "..", "edgecolor": "#5A918D"},
}

DEFAULT_DATASET_STYLE: dict[str, str] = {
    "color": "#999999",
    "hatch": "",
    "edgecolor": "#666666",
}


# --------------------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------------------

_LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
}


def escape_latex(text: str) -> str:
    """Escape the LaTeX special characters that occur in dataset captions."""
    for char, replacement in _LATEX_ESCAPES.items():
        text = text.replace(char, replacement)
    return text


def dataset_slug(dataset: str) -> str:
    """Filename/label-safe form of a dataset name, e.g. ``"ShareGPT4V" -> "sharegpt4v"``."""
    return dataset.lower().replace(" ", "_")
