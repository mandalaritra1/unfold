from __future__ import annotations

from pathlib import Path

from build_rho_gallery import build_html, parse_args


def main() -> None:
    args = parse_args()
    if args.root == Path(__file__).resolve().parent / "rho":
        args.root = Path(__file__).resolve().parent / "mass"
    if args.output is None:
        args.output = args.root / "index.html"
    build_html(
        args.root.resolve(),
        args.output.resolve(),
        args.dpi,
        page_title="Jet Mass Plot gallery",
        hero_title="Jet Mass Plot gallery",
        hero_copy="Quick review of plots stored in <code>/unfold/output/mass/</code>",
    )


if __name__ == "__main__":
    main()
