from __future__ import annotations

import argparse
import html
import subprocess
from collections import defaultdict
from pathlib import Path


PDF_DPI = 144
PREVIEW_DIRNAME = "_previews"
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
SKIP_DIR_NAMES = {PREVIEW_DIRNAME, ".ipynb_checkpoints"}
FOLDER_ORDER = {
    ".": 0,
    "unfold": 1,
    "uncertainties": 2,
    "data_mc": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML gallery for rho outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "rho",
        help="Folder containing saved rho outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <root>/index.html.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=PDF_DPI,
        help="DPI used for first-page PDF previews.",
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def nice_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def ensure_pdf_preview(pdf_path: Path, root: Path, dpi: int) -> Path:
    preview_root = root / PREVIEW_DIRNAME
    preview_path = (preview_root / pdf_path.relative_to(root)).with_suffix(".png")
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    if preview_path.exists() and preview_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        return preview_path

    output_prefix = preview_path.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            str(dpi),
            str(pdf_path),
            str(output_prefix),
        ],
        check=True,
    )
    return preview_path


def collect_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        if any(part.startswith(".") and part not in {".", ".."} for part in rel_parts):
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        files.append(path)
    return files


def sort_group_key(item: tuple[str, list[Path]]) -> tuple[int, str]:
    folder = item[0]
    return (FOLDER_ORDER.get(folder, 99), folder)


def card_html(path: Path, root: Path, dpi: int) -> str:
    title = html.escape(nice_name(path))
    label = html.escape(path.name)
    href = html.escape(rel(path, root))
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        preview = ensure_pdf_preview(path, root, dpi)
        preview_src = html.escape(rel(preview, root))
        media = (
            f'<a href="{href}" target="_blank" rel="noreferrer">'
            f'<img src="{preview_src}" alt="{title} preview" loading="lazy"></a>'
        )
        open_label = "Open PDF"
    else:
        media = (
            f'<a href="{href}" target="_blank" rel="noreferrer">'
            f'<img src="{href}" alt="{title} preview" loading="lazy"></a>'
        )
        open_label = "Open Image"

    return f"""
    <article class="card" data-folder="{html.escape(path.parent.relative_to(root).as_posix() or '.')}" data-name="{html.escape(path.name.lower())}">
      <div class="card-header">
        <div>
          <div class="card-title">{title}</div>
          <div class="card-meta">{label}</div>
        </div>
        <div class="card-type">{suffix[1:]}</div>
      </div>
      {media}
      <div class="card-actions">
        <a class="btn" href="{href}" target="_blank" rel="noreferrer">{open_label}</a>
        <a class="btn" href="{href}" download>Download</a>
      </div>
    </article>
    """


def build_html(root: Path, output: Path, dpi: int) -> None:
    files = collect_files(root)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        folder = path.parent.relative_to(root).as_posix() or "."
        groups[folder].append(path)

    sections = []
    for folder, items in sorted(groups.items(), key=sort_group_key):
        title = "root" if folder == "." else folder
        cards = "".join(card_html(item, root, dpi) for item in items)
        sections.append(
            f"""
            <section class="section" data-folder="{html.escape(folder)}">
              <div class="section-header">
                <h2>{html.escape(title)}</h2>
                <span>{len(items)} files</span>
              </div>
              <div class="grid">
                {cards}
              </div>
            </section>
            """
        )

    template = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>rho plot gallery</title>
  <style>
    :root {{
      --bg: #10151d;
      --bg-deep: #0b1017;
      --panel: rgba(24, 31, 42, 0.92);
      --panel-strong: rgba(31, 40, 54, 0.98);
      --ink: #edf1f7;
      --muted: #aab4c3;
      --line: rgba(153, 175, 201, 0.18);
      --accent: #ff8a57;
      --accent-soft: rgba(255, 138, 87, 0.14);
      --gold: #f0ba58;
      --sage: #7fc5aa;
      --rose: #f28aa5;
      --sky: #8cb7ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(255, 138, 87, 0.18), transparent 24rem),
        radial-gradient(circle at top left, rgba(140, 183, 255, 0.12), transparent 22rem),
        linear-gradient(180deg, #17202b 0%, var(--bg) 45%, var(--bg-deep) 100%);
    }}
    .shell {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      position: sticky;
      top: 0;
      z-index: 10;
      margin: 0 0 24px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
      background:
        linear-gradient(135deg, rgba(27, 35, 48, 0.92), rgba(18, 24, 34, 0.88));
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.28);
      border-radius: 0 0 22px 22px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(1.8rem, 3vw, 2.8rem);
      font-weight: 700;
      color: #fff3ed;
    }}
    .hero-copy {{
      color: var(--muted);
      margin-bottom: 14px;
      max-width: 70rem;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(16rem, 1fr) auto;
      gap: 12px;
      align-items: center;
    }}
    input {{
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(14, 19, 27, 0.92);
      font: inherit;
      color: var(--ink);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }}
    .count {{
      color: var(--muted);
      font-size: 0.95rem;
      white-space: nowrap;
    }}
    .section {{
      margin: 30px 0 0;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }}
    .section-header h2 {{
      margin: 0;
      font-size: 1.2rem;
      text-transform: capitalize;
      color: #e7edf8;
    }}
    .section-header span {{
      color: #d2d9e5;
      font-size: 0.9rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
    }}
    .card {{
      background:
        linear-gradient(180deg, rgba(30, 39, 54, 0.98), var(--panel));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 16px 34px rgba(0, 0, 0, 0.22);
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }}
    .card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 20px 42px rgba(0, 0, 0, 0.34);
      border-color: rgba(255, 138, 87, 0.38);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .card-title {{
      font-size: 1.05rem;
      font-weight: 700;
      line-height: 1.2;
    }}
    .card-meta {{
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .card-type {{
      color: #f6ead8;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: linear-gradient(135deg, rgba(240, 186, 88, 0.22), rgba(255, 138, 87, 0.16));
      border: 1px solid rgba(255, 138, 87, 0.18);
      border-radius: 999px;
      padding: 4px 9px;
      height: fit-content;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }}
    .card-actions {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    .btn {{
      text-decoration: none;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.03));
    }}
    .btn:hover {{
      border-color: var(--accent);
      color: var(--accent);
      background: linear-gradient(180deg, rgba(255, 138, 87, 0.18), rgba(255, 138, 87, 0.09));
    }}
    .section[data-folder="."] .section-header h2 {{ color: var(--accent); }}
    .section[data-folder="unfold"] .section-header h2 {{ color: var(--rose); }}
    .section[data-folder="uncertainties"] .section-header h2 {{ color: var(--sky); }}
    .section[data-folder="data_mc"] .section-header h2 {{ color: var(--sage); }}
    .hidden {{
      display: none !important;
    }}
    @media (max-width: 800px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <h1>rho plot gallery</h1>
      <div class="hero-copy">
        Static review view for plots already written under <code>{html.escape(str(root))}</code>.
        PDF previews are cached in <code>{html.escape(str(root / PREVIEW_DIRNAME))}</code>.
      </div>
      <div class="controls">
        <input id="filter" type="search" placeholder="Filter by file name or folder">
        <div class="count"><span id="visible-count">{len(files)}</span> / {len(files)} visible</div>
      </div>
    </div>
    {''.join(sections)}
  </div>
  <script>
    const cards = Array.from(document.querySelectorAll('.card'));
    const sections = Array.from(document.querySelectorAll('.section'));
    const count = document.getElementById('visible-count');
    const filter = document.getElementById('filter');

    function update() {{
      const term = filter.value.trim().toLowerCase();
      let visible = 0;
      for (const card of cards) {{
        const text = `${{card.dataset.folder}} ${{card.dataset.name}}`;
        const show = !term || text.includes(term);
        card.classList.toggle('hidden', !show);
        if (show) visible += 1;
      }}
      for (const section of sections) {{
        const hasVisible = section.querySelector('.card:not(.hidden)');
        section.classList.toggle('hidden', !hasVisible);
      }}
      count.textContent = String(visible);
    }}

    filter.addEventListener('input', update);
    update();
  </script>
</body>
</html>
"""

    output.write_text(template, encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "index.html"
    build_html(root=root, output=output, dpi=args.dpi)


if __name__ == "__main__":
    main()
