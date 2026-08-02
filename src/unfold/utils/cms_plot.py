"""Shared CMS-label plot helpers usable from any plotting script.

Kept free of heavy imports (no ROOT, no unfold.tools) so standalone scripts
can import it cheaply.
"""

import datetime
import os
import subprocess
from pathlib import Path

# Every saved figure is also written once per flavor (see
# save_cms_label_flavors): the canonical file carries the label the plot was
# drawn with; the others land in <dir>/<Flavor>/<name>.
DEFAULT_CMS_LABEL_FLAVORS = ("Internal", "Preliminary", "Private Work")

# Type sizes for the figures the PAS/paper includes at ~0.49\textwidth, where
# on-page type is roughly a third of the drawn size. The CMS style's defaults
# (labels 26, ticks and legend ~21.7 on the 10x10 canvas) leave the legends
# and tick numbers too small there, so these paths set the sizes explicitly.
# Import these rather than redefining them -- the values drifted apart while
# each script kept its own copy.
PUB_LABEL_FONTSIZE = 30
PUB_TICK_FONTSIZE = 26
PUB_LEGEND_FONTSIZE = 20
PUB_ANNOTATION_FONTSIZE = 22

_REPO_ROOT = Path(__file__).resolve().parents[3]


def apply_pub_fonts(ax, label_fontsize=PUB_LABEL_FONTSIZE,
                    tick_fontsize=PUB_TICK_FONTSIZE):
    """Boost an axes' existing axis labels and tick numbers to the PUB sizes.

    Convenience for scripts that set their labels before styling; the sizes
    are applied to whatever text ``ax`` already carries.
    """
    ax.xaxis.label.set_fontsize(label_fontsize)
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)


def stamp_figure(fig, inputs="n/a", repo="unfold", date=None):
    """Stamp a small provenance line bottom-right, outside the axes.

    ``YYYY-MM-DD  |  <repo> <git describe>  |  inputs: <tag>``, so a working
    plot found on disk months later can be traced to the code that drew it.
    Set ``UNFOLD_NO_STAMP=1`` to suppress it for publication-final figures.
    """
    if os.environ.get("UNFOLD_NO_STAMP"):
        return None
    try:
        version = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, cwd=_REPO_ROOT, timeout=10,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        version = "unknown"
    # Constrained layout otherwise packs the axes flush to the bottom edge and
    # the stamp lands on top of the x-axis label; reserve a sliver for it.
    engine = fig.get_layout_engine()
    if engine is not None:
        try:
            engine.set(rect=(0, 0.028, 1, 0.972))
        except (AttributeError, TypeError, ValueError):
            pass
    stamp = date or datetime.date.today().isoformat()
    return fig.text(
        0.99, 0.006, f"{stamp}  |  {repo} {version}  |  inputs: {inputs}",
        ha="right", va="bottom", fontsize=7, color="0.45", family="monospace",
    )


def save_cms_label_flavors(fig, path, current_label,
                           flavors=DEFAULT_CMS_LABEL_FLAVORS, **savefig_kw):
    """Save ``fig`` at ``path``, then once per CMS-label flavor.

    The figure is drawn once; for each flavor the mplhep label text artists
    are swapped in place (plain substring replacement, so e.g.
    "Simulation Internal" -> "Simulation Private Work") and the figure
    re-saved under ``<path.parent>/<Flavor>/<path.name>`` (flavor dir without
    spaces), then restored.
    """
    import matplotlib.text as mtext

    path = Path(path)
    fig.savefig(path, **savefig_kw)
    if not current_label:
        return
    suffix_texts = {t: t.get_text() for t in fig.findobj(mtext.Text)
                    if current_label in t.get_text()}
    if not suffix_texts:
        return
    try:
        for flavor in flavors:
            if flavor == current_label:
                continue
            for t, original in suffix_texts.items():
                t.set_text(original.replace(current_label, flavor))
            out = path.parent / flavor.replace(" ", "") / path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, **savefig_kw)
    finally:
        for t, original in suffix_texts.items():
            t.set_text(original)
