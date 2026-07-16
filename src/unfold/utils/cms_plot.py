"""Shared CMS-label plot helpers usable from any plotting script.

Kept free of heavy imports (no ROOT, no unfold.tools) so standalone scripts
can import it cheaply.
"""

from pathlib import Path

# Every saved figure is also written once per flavor (see
# save_cms_label_flavors): the canonical file carries the label the plot was
# drawn with; the others land in <dir>/<Flavor>/<name>.
DEFAULT_CMS_LABEL_FLAVORS = ("Internal", "Preliminary", "Private Work")


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
