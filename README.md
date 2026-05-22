Use the latest shared unfolder implementation in `src/unfold/tools/unfolder_core.py`.

## Documentation

- Core class reference: [docs/Unfolder_core_class_reference.md](docs/Unfolder_core_class_reference.md)

For the rho workflow:

- Run `notebooks/unfolder_v4_rho.ipynb` to produce tagged rho outputs under `outputs/rho/original/` and `outputs/rho/fixed_jec/`.
- Review saved plots in `notebooks/rho_review.ipynb`.
- Build static scrollable galleries with `python3 outputs/build_rho_gallery.py --root outputs/rho/original` and `python3 outputs/build_rho_gallery.py --root outputs/rho/fixed_jec`.
- Open `outputs/rho/original/index.html` or `outputs/rho/fixed_jec/index.html` in a browser when you want a fast overview of many plots.
