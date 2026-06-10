from pathlib import Path
import html
import subprocess
from collections import defaultdict

ROOT = Path(__file__).resolve().parent  # outputs/rho
OUT = ROOT / "index.html"

# Where auto-generated preview PNGs go
PREVIEW_DIR = ROOT / "_previews"  # keeps your real folders clean

PDF_DPI = 150  # 120-200 is a good range for thumbnails

def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()

def nice_title(filename: str) -> str:
    # Tweak freely (e.g. replace "__" etc.)
    return filename.replace("_", " ")

def ensure_pdf_preview(pdf_path: Path) -> Path:
    """
    Generate a PNG preview (first page) for a PDF using pdftoppm.
    Caches it under ROOT/_previews/<same relative folder>/<stem>.png
    """
    pdf_rel = pdf_path.relative_to(ROOT)
    out_png = (PREVIEW_DIR / pdf_rel).with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # If cached preview exists and is newer than pdf, reuse it
    if out_png.exists() and out_png.stat().st_mtime >= pdf_path.stat().st_mtime:
        return out_png

    # pdftoppm wants output prefix WITHOUT extension
    out_prefix = out_png.with_suffix("")  # /.../_previews/foo/bar/myplot
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            str(PDF_DPI),
            str(pdf_path),
            str(out_prefix),
        ],
        check=True,
    )
    return out_png

# Collect items: each file is an item (no pairing required)
pdf_files = sorted(ROOT.rglob("*.pdf"))
png_files = sorted([p for p in ROOT.rglob("*.png") if PREVIEW_DIR not in p.parents])

# Optional helper maps (only for the "bonus" pdf button on png cards)
pdf_by_dir_stem = {(p.parent, p.stem): p for p in pdf_files}

# Group items by folder
groups = defaultdict(list)
for p in pdf_files + png_files:
    folder = p.parent.relative_to(ROOT).as_posix()
    groups[folder].append(p)

def make_card(p: Path) -> str:
    title = html.escape(nice_title(p.name))

    if p.suffix.lower() == ".pdf":
        preview_png = ensure_pdf_preview(p)
        preview_html = (
            f'<a href="{rel(p)}" target="_blank" rel="noreferrer">'
            f'<img src="{rel(preview_png)}" alt="{title} preview"></a>'
        )
        buttons = f'<a class="btn" href="{rel(p)}" download>Download PDF</a>'

    else:  # .png
        preview_html = (
            f'<a href="{rel(p)}" target="_blank" rel="noreferrer">'
            f'<img src="{rel(p)}" alt="{title} preview"></a>'
        )
        buttons_list = [f'<a class="btn" href="{rel(p)}" download>Download PNG</a>']

        # Bonus: if a same-stem PDF exists in same folder, add a PDF download button
        maybe_pdf = pdf_by_dir_stem.get((p.parent, p.stem))
        if maybe_pdf:
            buttons_list.insert(0, f'<a class="btn" href="{rel(maybe_pdf)}" download>Download PDF</a>')

        buttons = "".join(buttons_list)

    return f"""
      <div class="card">
        <p class="title">{title}</p>
        {preview_html}
        <div class="btns">{buttons}</div>
      </div>
    """

sections = []
for folder, items in sorted(groups.items(), key=lambda x: x[0]):
    header = "root" if folder in ("", ".") else folder
    cards = "\n".join(make_card(p) for p in sorted(items, key=lambda x: (x.suffix, x.name)))
    sections.append(f"""
    <h2>{html.escape(header)}</h2>
    <div class="grid">
      {cards}
    </div>
    """)

template = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Plot Gallery</title>
  <style>
    body {{ font-family: system-ui, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin: 28px 0 12px; font-size: 16px; color: #333; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 12px; }}
    .title {{ font-weight: 650; margin: 0 0 10px; }}
    img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #eee; }}
    .btns {{ display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }}
    a.btn {{
      display: inline-block; padding: 8px 10px; border-radius: 10px;
      border: 1px solid #ccc; text-decoration: none; color: #111;
    }}
    a.btn:hover {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Plot Gallery</h1>
  <div style="color:#666;font-size:12px;margin-bottom:16px;">
    Previews for PDFs are cached in: {html.escape(rel(PREVIEW_DIR))}
  </div>
  {"".join(sections)}
</body>
</html>
"""

OUT.write_text(template, encoding="utf-8")
print(f"Wrote {OUT}")
