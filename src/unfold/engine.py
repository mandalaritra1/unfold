"""The unfolding engine.

``Unfolder`` takes an ``UnfoldInputs`` (see ``inputs.py``) and a spec, runs
TUnfold (or RooUnfold's iterative Bayes) for the nominal input and every
systematic variation, propagates the statistical uncertainty (jackknife
replicas when the inputs carry them, otherwise TUnfold's covariances),
normalizes each pT slice, combines the systematics, and evaluates the
bottom-line test.  Plotting lives in ``plots.py``; the data products
(``save_*``) are written by this class.

Attribute conventions
---------------------
Everything from ``UnfoldInputs`` is copied onto the instance under the same
name, so ``self.mosaic_dict``, ``self.misses_2d`` etc. mean the same thing in
the loaders, here and in the plots.  Results are added by ``run``:

``y_unf, ye_unf``            unfolded spectrum and selected statistical error (flat gen)
``y_meas, y_true, x_folded`` fake-corrected data, MC truth prior, refolded result
``y_unf_dict``               unfolded spectrum per systematic
``cov_np, cov_data_np, cov_uncorr_np``  selected total / input-stat / matrix-stat covariances
``normalized_results``       per pT slice: normalized unfolded, truth, stat and total bands
``normalized_systematics``   per pT slice: normalized unfolded per systematic
``norm_cov_*``               stat covariances of the normalized result (jacobian propagation)
"""

from __future__ import annotations

from array import array
from dataclasses import fields
from pathlib import Path
import pickle as pkl
import re

import hist
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from scipy.stats import chi2 as scipy_chi2

from unfold.cms_plot import DEFAULT_CMS_LABEL_FLAVORS, save_cms_label_flavors, stamp_figure
from unfold.histmath import merge_mass_flat, unflatten_gen_by_pt
from unfold.inputs import UnfoldInputs, scale_covariance_by_fake_survival
from unfold import systematics as systnames


def _declare_open_l():
    """Expose TUnfold's protected AddRegularisationCondition and scan helpers.

    The subclass is needed for the ratio-curvature regularization (the public
    RegularizeCurvature cannot express the exact (1/m0, -2/m1, 1/m2) row).
    The scan helpers exist because PyROOT cannot pass TGraph**/TSpline**
    output arguments directly.
    """
    if not hasattr(ROOT, "TUnfoldDensityOpenL"):
        ROOT.gInterpreter.Declare(
            """
            class TUnfoldDensityOpenL : public TUnfoldDensity {
            public:
              using TUnfoldDensity::TUnfoldDensity;
              using TUnfoldDensity::AddRegularisationCondition;
            };

            struct UnfoldLcurveScan {
              TGraph* lcurve = nullptr;
              TSpline* logTauX = nullptr;
              TSpline* logTauY = nullptr;
              TSpline* curvature = nullptr;
              Int_t iBest = -1;
            };
            UnfoldLcurveScan RunUnfoldLcurveScan(TUnfold& u, Int_t nScan) {
              UnfoldLcurveScan r;
              r.iBest = u.ScanLcurve(nScan, 0., 0., &r.lcurve,
                                     &r.logTauX, &r.logTauY, &r.curvature);
              return r;
            }

            struct UnfoldSureScan {
              TGraph* logTauSURE = nullptr;
              TGraph* df_chi2A = nullptr;
              TGraph* lCurve = nullptr;
              Int_t iBest = -1;
            };
            UnfoldSureScan RunUnfoldSureScan(TUnfoldDensity& u, Int_t nScan,
                                             Double_t tauMin, Double_t tauMax) {
              UnfoldSureScan r;
              r.iBest = u.ScanSURE(nScan, tauMin, tauMax, &r.logTauSURE,
                                   &r.df_chi2A, &r.lCurve);
              return r;
            }
            """
        )


def _graph_to_arrays(graph):
    n = graph.GetN()
    return (
        np.array([graph.GetPointX(i) for i in range(n)]),
        np.array([graph.GetPointY(i) for i in range(n)]),
    )


def _unused_merge_edges_below(edges, threshold, *, tol=1e-9):
    """Collapse all bins below ``threshold`` into a single low-edge bin.

    Keeps the first (lowest) edge and every edge at or above ``threshold``,
    dropping the interior edges strictly below it. E.g. with threshold -2.5,
    ``[-10, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]`` becomes
    ``[-10, -2.5, -2, -1.5, -1, -0.5, 0]``. Idempotent when the binning is
    already merged at or above the threshold.
    """
    edges = [float(e) for e in edges]
    kept = [edges[0]]
    kept.extend(e for e in edges[1:] if e >= threshold - tol)
    return kept



class Unfolder:
    # Flavors every saved plot is replicated into (see _save_label_flavors);
    # override on an instance to restrict, e.g. u.cms_label_flavors = ()
    cms_label_flavors = DEFAULT_CMS_LABEL_FLAVORS

    def __init__(self, inputs: UnfoldInputs, spec, groomed, *, cms_label="Internal",
                 lumi=138.0, com=13.0, closure=False, herwig_closure=False):
        for f in fields(inputs):
            setattr(self, f.name, getattr(inputs, f.name))
        self._uses_prepared_inputs = inputs.uses_prepared_inputs
        self.inputs = inputs
        self.mosaic = self.mosaic_dict["nominal"]          # nominal response, (n_reco, n_gen)
        self.M_np_2d = self.M_np_2d_dict.get("nominal")
        b = inputs.binning
        self.bins = b
        self.pt_edges = list(b.pt_edges)
        self.edges = list(b.reco_edges)
        self.edges_gen = list(b.gen_edges)
        self.reco_edges_by_pt = [list(e) for e in b.reco_edges_by_pt]
        self.gen_edges_by_pt = [list(e) for e in b.gen_edges_by_pt]

        self.spec = spec
        self.reco_axis = spec.reco_axis
        self.gen_axis = spec.gen_axis
        self.groomed = groomed
        self.cms_label = cms_label
        self.lumi = float(lumi)
        self.com = float(com)
        self.stat_propagation = spec.stat_propagation
        self.regularization = spec.regularization
        self.tau = spec.tau              # None -> L-curve scan on the nominal data unfold sets it
        self.method = spec.method        # "tunfold" or "roounfold_bayes"
        self.n_iter = spec.n_iter
        # closure: unfold the MC reco spectrum; herwig_closure: unfold the HERWIG reco spectrum
        self.closure = closure
        self.herwig_closure = herwig_closure
        self.y_unf_dict = {}
        self.ye_unf_dict = {}
        self.y_unf_jk_input_list = []
        self.y_unf_jk_matrix_list = []
        self._ensure_output_dirs()

    def run(self, *, statistics=None):
        """Unfold nominal + systematics, then statistics, normalization, totals."""
        self._perform_unfold(systematic="nominal", closure=self.closure, herwig_closure=self.herwig_closure)
        for systematic in self.systematics:
            if systematic != "nominal":
                self._perform_unfold(systematic=systematic, closure=self.closure, herwig_closure=self.herwig_closure)
        if statistics is not None:
            # Pair-split replicas install the selected absolute covariances
            # and their exact, per-replica normalized counterparts here.
            statistics(self)
        if self.has_jackknife:
            self._compute_stat_unc()
        else:
            # Read the selected covariances (analytic unless installed above).
            self._compute_input_stat_unc_from_covariance()
        self._normalize_result()
        self._compute_total_systematic()
        return self

    # Pairs "<base>Up[_corr|_uncorr_YYYY]" with the matching Down key.
    _UPDOWN_KEY = systnames.UPDOWN_KEY

    def _ensure_output_dirs(self):
        output_dir = Path(self.spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Category subfolders (summary/, inputs/, response/, bottom_line/,
        # unfolded/, validation/, uncertainties/, data/) are created on demand by
        # _categorize_output; only the gallery preview mirror is pre-made here.
        (output_dir / "_previews").mkdir(parents=True, exist_ok=True)


    @property
    def _band_is_stat_only(self):
        """True when nothing but the statistical error feeds the total band.

        With no systematic variations unfolded and no model envelope,
        ``syst_unc`` is stat_unc by construction (_compute_total_systematic
        seeds the quadrature sum with it), so the "total" band would paint the
        same box as the stat band under a legend entry that promises
        systematics -- and both would repeat what the markers' error bars
        already show. The self-closure figures run stats-only: they draw NO
        filled band, and the stat uncertainty appears once, on the points.
        """
        return (set(getattr(self, "systematics", ["nominal"])) <= {"nominal"}
                and not getattr(self.spec, "model_envelope", False))


    def _categorize_output(self, stem):
        """Map a plot basename to (category_subfolder, descriptive_stem).

        Single source of truth for the per-mode output layout: every plot lands
        in a semantic subfolder (nothing dumped in the mode root) and the bare
        unfolded-result names (e.g. 'groomed_1') gain a descriptive prefix. The
        trailing '_<mode>_<idx>' panel index is normalized to '_<mode>_pt<idx>'
        ('ptall' for the inclusive -1 panel) so filenames are searchable.
        """
        def _ptnorm(name):
            return re.sub(
                r"_(groomed|ungroomed)_(-?\d+)$",
                lambda m: f"_{m.group(1)}_"
                + ("ptall" if int(m.group(2)) < 0 else f"pt{m.group(2)}"),
                name,
            )

        s = stem
        # bare unfolded result, e.g. 'groomed_1' / 'ungroomed_-1'
        if re.fullmatch(r"(?:un)?groomed_-?\d+", s):
            return "unfolded", _ptnorm("unfolded_" + s)
        # per-mode result summaries, e.g. 'groomed_summary', 'groomed_summary_linear'
        m = re.fullmatch(r"(groomed|ungroomed)_summary(_linear|_ratio)?", s)
        if m:
            return "summary", f"unfolded_summary{m.group(2) or ''}_{m.group(1)}"
        # prefix -> (category, rename); first match wins, order matters
        rules = [
            ("bottom_line_chi2",         "summary",       None),
            ("bottom_line",              "bottom_line",   None),
            ("purity_stability",         "response",      None),
            ("response_",                "response",      lambda n: n.replace("response_", "response_matrix_", 1)),
            ("fakerates",                "response",      lambda n: n.replace("fakerates", "fakes_misses", 1)),
            ("input_",                   "inputs",        lambda n: n.replace("input_", "input_data_mc_", 1)),
            ("herwig_pythia_comparison", "unfolded",      None),
            ("herwig_closure_unc",       "validation",    None),
            ("herwig_bias_test",         "validation",    None),
            ("closure_",                 "validation",    None),
            ("folded_",                  "validation",    None),
            ("jackknife_convergence",    "validation",    None),
            ("jk_inputs",                "validation",    None),
            ("jk_outputs",               "validation",    None),
            ("L_matrix",                 "validation",    None),
            ("lcurve",                   "validation",    None),
            ("correlation",              "unfolded",      None),
            ("unfolded_unrolled_2d",     "unfolded",      None),
            ("unfolded_basic",           "unfolded",      None),
            ("unfolded_2d",              "data",          None),
            ("uncertainty_summary_2d",   "data",          None),
            ("normalized_covariance",    "data",          None),
            ("stat_fraction",            "uncertainties", None),
            ("summary_grouped",          "uncertainties", None),
            ("summary_linear",           "uncertainties", None),
            ("summary_",                 "uncertainties", None),
            ("nominal_minus_",           "uncertainties", None),
            ("heatmap",                  "uncertainties", None),
        ]
        for prefix, cat, fn in rules:
            if s.startswith(prefix):
                return cat, _ptnorm(fn(s) if fn else s)
        # remaining '<SYST>_<mode>_<idx>' plots (JES/JMS/ElectronSF/herwig/...)
        if re.search(r"_(groomed|ungroomed)_-?\d+$", s):
            return "uncertainties", _ptnorm(s)
        return "misc", _ptnorm(s)


    def _relocate_output(self, save_path):
        """Rewrite a legacy save path into the categorized layout."""
        p = Path(save_path)
        cat, stem = self._categorize_output(p.stem)
        return Path(self.spec.output_dir) / cat / f"{stem}{p.suffix}"


    def _finalize_plot(self, save_path=None, show=True, fig=None):
        if save_path is not None:
            path = self._relocate_output(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            target_fig = fig if fig is not None else plt.gcf()
            stamp_figure(target_fig, inputs=Path(self.spec.input_dir).name if self.spec.input_dir else "prepared")
            self._save_label_flavors(target_fig, path,
                                     bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        else:
            plt.close(fig if fig is not None else plt.gcf())


    def _save_label_flavors(self, fig, path, **savefig_kw):
        save_cms_label_flavors(fig, path, self.cms_label,
                               flavors=self.cms_label_flavors, **savefig_kw)


    def _cms_extra_label(self):
        return f" {self.cms_label}" if self.cms_label and not self.cms_label.startswith(" ") else self.cms_label


    @staticmethod
    def _as_int_when_whole(value):
        """138.0 -> 138 so the CMS header shows integers, 59.7 stays 59.7."""
        return int(value) if float(value).is_integer() else value


    @staticmethod
    def _subset_fraction_unc(numer, other, numer_var, other_var):
        """MC-stat error of f = numer/(numer+other) for disjoint subsets."""
        total = numer + other
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(
                total > 0,
                (other**2 * numer_var + numer**2 * other_var) / total**4,
                0.0,
            )
        return np.sqrt(np.clip(var, 0.0, None))


    def _lumi_label(self):
        return self._as_int_when_whole(self.lumi)


    def _com_label(self):
        return self._as_int_when_whole(self.com)


    def _reported_pt_indices(self):
        return range(
            getattr(self, "first_reported_pt_bin", 0),
            len(self.pt_edges) - 1,
        )


    def _output_panel_index(self, i_pt):
        """Return a zero-based reported-slice index for plot filenames.

        Legacy rho inputs commonly reserve pT slice zero as an unreported
        migration sink and therefore begin at ``first_reported_pt_bin == 1``.
        Prepared pair-split inputs instead report their physical first slice at
        zero.  Deriving names from the explicit boundary keeps both layouts
        unambiguous (and avoids a misleading ``ptall`` name for pair-split).
        """
        return int(i_pt) - int(getattr(self, "first_reported_pt_bin", 0))


    def _summary_pt_indices(self):
        """Reported pT slices for summary plots.

        Some legacy inputs keep a 0--200 GeV migration sink at index zero, but
        pair-split inputs start their physical measurement at index zero.  The
        explicit ``first_reported_pt_bin`` is the only source of truth.
        """
        return self._reported_pt_indices()


    def _reported_matrix_view(self, matrix):
        first_pt_bin = getattr(self, "first_reported_pt_bin", 0)
        reco_offset = sum(
            len(edges) - 1 for edges in self.reco_edges_by_pt[:first_pt_bin]
        )
        gen_offset = sum(
            len(edges) - 1 for edges in self.gen_edges_by_pt[:first_pt_bin]
        )
        return (
            np.asarray(matrix)[reco_offset:, gen_offset:],
            self.reco_edges_by_pt[first_pt_bin:],
            self.gen_edges_by_pt[first_pt_bin:],
            self.pt_edges[first_pt_bin:],
        )


    def _has_systematic(self, *prefixes):
        lowered = tuple(prefix.lower() for prefix in prefixes)
        return any(
            systematic.lower().startswith(lowered)
            for systematic in self.systematics
            if systematic != "nominal"
        )


    def _observable_label(self):
        return self.spec.x_label_groomed if self.groomed else self.spec.x_label_ungroomed


    @staticmethod
    def _format_pt_edge(value):
        """Format integral pT edges without the producer's trailing ``.0``."""
        value = float(value)
        if np.isclose(value, round(value), rtol=0.0, atol=1.0e-9):
            return str(int(round(value)))
        return f"{value:g}"


    def _pt_bin_label(self, pt_bin):
        low = self._format_pt_edge(pt_bin[0])
        if pt_bin[1] == float("inf") or pt_bin[1] > 100000:
            return f"{low}–∞"
        return f"{low}–{self._format_pt_edge(pt_bin[1])}"


    def _observable_short_label(self):
        return self.spec.short_label_groomed if self.groomed else self.spec.short_label_ungroomed


    def _observable_xlim(self, i_pt=None):
        if i_pt is not None:
            edges = self.gen_edges_by_pt[i_pt]
            upper = float(edges[-1] if self.spec.name == "rho" else edges[-2])
            # Per-pT display floor (falls back to the mode's uniform
            # xlim_lower when the spec sets no per-pT floors).
            return (self._bl_shown_floors()[i_pt], upper)
        edge_index = -1 if self.spec.name == "rho" else -2
        upper = max(float(edges[edge_index]) for edges in self.gen_edges_by_pt)
        lower = self.spec.xlim_lower_groomed if self.groomed else self.spec.xlim_lower_ungroomed
        return (lower, upper)


    def _normalized_ylabel(self):
        return self.spec.normalized_ylabel


    def _histogram_keys(self):
        return self.spec.hist_keys_groomed if self.groomed else self.spec.hist_keys_ungroomed


    def _apply_background_subtraction(self, meas_flat, closure, herwig_closure):
        """Remove the non-DY background from the measured spectrum.

        Applied BEFORE the fake correction: the fake fraction is derived from DY
        simulation as fakes/(matched+fakes) and describes what fraction of the
        *DY* reco yield originates outside the fiducial gen phase space. ttbar,
        single top and diboson events are not DY at all, so they must leave the
        spectrum first; scaling them by (1-f) instead would attribute a DY
        migration property to a foreign process.

        Closure tests unfold simulation through its own response and contain no
        background by construction, so they are returned unchanged.
        """
        if closure or herwig_closure:
            return meas_flat
        if getattr(self, "bkg_2d", None) is None:
            return meas_flat
        subtracted = np.asarray(meas_flat, dtype=float) - np.asarray(self.bkg_2d, dtype=float)
        return np.clip(subtracted, 0.0, None)


    def _add_background_variance(self, variances, closure, herwig_closure):
        """Add the background MC-stat variance to the input variance.

        Data and simulated background are independent, so the variances add.
        No-op when nothing was subtracted, or when the background pkl carries no
        sumw2 (in which case only the data statistical term is propagated).
        """
        if closure or herwig_closure:
            return variances
        if getattr(self, "bkg_var_2d", None) is None:
            return variances
        return np.asarray(variances, dtype=float) + np.asarray(self.bkg_var_2d, dtype=float)


    def _compute_stat_unc(self):
        for i in range(10):
            meas_flat = self.mosaic_2d_jk_list[i]
            self._perform_unfold(
                systematic='nominal',
                closure=self.closure,
                herwig_closure=self.herwig_closure,
                meas_flat=meas_flat,
                do_jk=True,
                jk_target="input",
            )

        for i in range(10):
            resp_np = self.mosaic_jk_list[i]
            self._perform_unfold(
                systematic='nominal',
                closure=self.closure,
                herwig_closure=self.herwig_closure,
                resp_np=resp_np,
                do_jk=True,
                jk_target="matrix",
            )

        # Delete-one-tenth (grouped) jackknife with g=10 groups. The correct SE
        # on the population std of the replicas (np.std, ddof=0) is sqrt(g-1);
        # var_jack = (g-1)/g * sum((theta_i - mean)^2) = (g-1) * s_pop^2. The old
        # sqrt(10/9) was a Bessel-style factor that under-covered by ~2.85x.
        jk_scale = np.sqrt(9.0)
        input_std = jk_scale * np.std(self.y_unf_jk_input_list, axis=0)
        matrix_std = jk_scale * np.std(self.y_unf_jk_matrix_list, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.input_stat_unc_frac = np.abs(
                np.divide(input_std, self.y_unf, out=np.zeros_like(input_std), where=self.y_unf != 0)
            )
            self.matrix_stat_unc_frac = np.abs(
                np.divide(matrix_std, self.y_unf, out=np.zeros_like(matrix_std), where=self.y_unf != 0)
            )

        self.stat_unc_frac = np.sqrt(self.input_stat_unc_frac**2 + self.matrix_stat_unc_frac**2)
        self.input_stat_unc_pt_binned = unflatten_gen_by_pt(self.input_stat_unc_frac, self.gen_edges_by_pt)
        self.matrix_stat_unc_pt_binned = unflatten_gen_by_pt(self.matrix_stat_unc_frac, self.gen_edges_by_pt)
        self.stat_unc_pt_binned = unflatten_gen_by_pt(self.stat_unc_frac, self.gen_edges_by_pt)


    def _select_measured_spectrum(self, closure, herwig_closure, meas_flat):
        if meas_flat is None:
            if closure:
                meas_flat = self.mosaic.sum(axis=1)
            else:
                meas_flat = self.mosaic_2d
            if herwig_closure:
                meas_flat = self.mosaic_herwig_2d
        return meas_flat


    def _apply_fake_correction(self, meas_flat, systematic, closure, herwig_closure):
        if closure or herwig_closure:
            return meas_flat

        fake_fraction = self.fake_fraction_2d
        if systematic in {"herwigUp", "herwigDown"} and self.fake_fraction_2d_herwig is not None:
            fake_fraction = self.fake_fraction_2d_herwig
        elif getattr(self, "_uses_prepared_inputs", False):
            if systematic not in self.fake_fraction_2d_dict:
                raise ValueError(
                    f"Prepared fake fraction is missing systematic {systematic!r}"
                )
            fake_fraction = self.fake_fraction_2d_dict[systematic]

        corrected = np.asarray(meas_flat, dtype=float) * (1.0 - fake_fraction)
        return np.clip(corrected, 0.0, None)


    def _fake_survival(self, systematic):
        """1 - fake fraction of the response used for ``systematic``."""
        if systematic in {"herwigUp", "herwigDown"}:
            return 1.0 - np.asarray(self.fake_fraction_2d_herwig, dtype=float)
        if self.fake_survival_dict is not None:
            if systematic not in self.fake_survival_dict:
                raise ValueError(f"Prepared fake survival is missing systematic {systematic!r}")
            return self.fake_survival_dict[systematic]
        return 1.0 - np.asarray(self.fake_fraction_2d, dtype=float)

    def _build_root_binning(self):
        truth_root = ROOT.TUnfoldBinning("truth")
        reco_root = ROOT.TUnfoldBinning("reco")

        truth_signal = truth_root.AddBinning("signal")
        reco_primary = reco_root.AddBinning("primary")

        for i, edges in enumerate(self.gen_edges_by_pt):
            truth_node = truth_signal.AddBinning(f"pt{i}")
            truth_node.AddAxis("mass", len(edges) - 1, array("d", edges), False, False)

        for i, edges in enumerate(self.reco_edges_by_pt):
            reco_node = reco_primary.AddBinning(f"pt{i}")
            reco_node.AddAxis("mass", len(edges) - 1, array("d", edges), False, False)

        return truth_root, reco_root


    def _fill_root_histogram(self, hist, values, variances=None):
        for index, value in enumerate(values, 1):
            hist.SetBinContent(index, float(value))
            if variances is not None:
                hist.SetBinError(index, float(np.sqrt(max(variances[index - 1], 0.0))))


    @staticmethod
    def _root_covariance_histogram(covariance, name="hInputCovariance"):
        """Convert a flattened covariance into TUnfold's TH2 input format."""
        covariance = np.asarray(covariance, dtype=float)
        n_bins = covariance.shape[0]
        h_covariance = ROOT.TH2D(name, name, n_bins, 0.0, float(n_bins),
                                 n_bins, 0.0, float(n_bins))
        for i in range(n_bins):
            for j in range(n_bins):
                h_covariance.SetBinContent(i + 1, j + 1, covariance[i, j])
        return h_covariance


    def _fill_response_histogram(
        self,
        h_resp,
        resp_np,
        misses,
        *,
        include_bin_errors=True,
        resp_var=None,
        misses_var=None,
    ):
        """Fill the migration matrix + miss row, with optional MC-stat errors.

        When ``resp_var``/``misses_var`` (sumw2 from the pkl hist objects) are
        given, bin errors are set to sqrt(var). Without them ROOT defaults
        GetBinError to sqrt(weighted content), which overestimates the MC-stat
        error of weighted histograms — so pass them whenever available.
        """
        n_reco, n_true = resp_np.shape
        for i_reco in range(n_reco):
            for j_true in range(n_true):
                h_resp.SetBinContent(j_true + 1, i_reco + 1, resp_np[i_reco, j_true])
                if not include_bin_errors:
                    h_resp.SetBinError(j_true + 1, i_reco + 1, 0.0)
                elif resp_var is not None:
                    h_resp.SetBinError(
                        j_true + 1, i_reco + 1,
                        float(np.sqrt(max(resp_var[i_reco, j_true], 0.0))),
                    )
        for j_true in range(n_true):
            h_resp.SetBinContent(j_true + 1, 0, misses[j_true])
            if not include_bin_errors:
                h_resp.SetBinError(j_true + 1, 0, 0.0)
            elif misses_var is not None:
                h_resp.SetBinError(
                    j_true + 1, 0, float(np.sqrt(max(misses_var[j_true], 0.0)))
                )


    def _add_ratio_curvature_conditions(self, unfold, prior_flat):
        """Register curvature-of-ratio regularization rows.

        For interior gen bins of each pT slice adds the row
        (1/m0, -2/m1, 1/m2) so that any spectrum proportional to the prior m
        (the nominal MC truth) has exactly zero penalty; only shape deviations
        from the prior are smoothed. No conditions cross pT-slice boundaries.
        Validated in scripts/studies/study_regularization_rho.py: exact self-closure
        at any tau, <1% added HERWIG-closure bias at the L-curve tau.
        """
        offset = 0
        for edges in self.gen_edges_by_pt:
            nbins = len(edges) - 1
            for k in range(1, nbins - 1):
                j0, j1, j2 = offset + k - 1, offset + k, offset + k + 1
                m0, m1, m2 = prior_flat[j0], prior_flat[j1], prior_flat[j2]
                if min(m0, m1, m2) <= 0:
                    continue
                unfold.AddRegularisationCondition(
                    j0 + 1, 1.0 / m0, j1 + 1, -2.0 / m1, j2 + 1, 1.0 / m2
                )
            offset += nbins


    def _curvature_regularization_ranges(self):
        """Inclusive flattened truth-bin ranges, one independent row set per pT."""
        ranges = []
        offset = 0
        for edges in self.gen_edges_by_pt:
            n_bins = len(edges) - 1
            if n_bins >= 3:
                ranges.append((offset + 1, offset + n_bins))
            offset += n_bins
        return ranges


    def _add_curvature_regularization(self, unfold):
        """Register public TUnfold curvature rows without crossing pT slices."""
        for first_bin, last_bin in self._curvature_regularization_ranges():
            unfold.RegularizeBins(
                first_bin,
                1,
                last_bin - first_bin + 1,
                ROOT.TUnfold.kRegModeCurvature,
            )


    def _store_covariances(self, unfold, systematic):
        if systematic == "nominal":
            self.cov = unfold.GetEmatrixTotal("cov", "Covariance Matrix")
            self.cov_uncorr = unfold.GetEmatrixSysUncorr(
                "cov_uncorr",
                "Covariance Matrix from Uncorrelated Uncertainties",
            )
            self.cov_uncorr_data = unfold.GetEmatrixInput(
                "cov_uncorr_data",
                "Covariance Matrix from Stat Uncertainties of Input Data",
            )
            self.cov_total = unfold.GetEmatrixTotal("total", "Cov")

            _, n_true = self.mosaic.shape
            self.cov_np = np.zeros((n_true, n_true))
            self.cov_uncorr_np = np.zeros((n_true, n_true))
            self.cov_data_np = np.zeros((n_true, n_true))
            for i in range(1, n_true + 1):
                for j in range(1, n_true + 1):
                    self.cov_np[i - 1, j - 1] = self.cov.GetBinContent(i, j)
                    self.cov_uncorr_np[i - 1, j - 1] = self.cov_uncorr.GetBinContent(i, j)
                    self.cov_data_np[i - 1, j - 1] = self.cov_uncorr_data.GetBinContent(i, j)

            # Reco-space hat matrix H = K J for the bottom-line N_dof (TWiki):
            # K is the smearing/probability matrix (reco x gen), J = dx/dy the
            # error-propagation matrix (gen x reco).  Both are exact TUnfold
            # linear-response outputs; must be read while ``unfold`` is alive
            # (see _store_unfold_result).  useAxisBinning=False -> flat global
            # bins.  The TWiki N_dof is the effective rank of K J J^T K^T; for
            # the oblique reco-space projection K J that rank equals the trace
            # of H (the standard effective number of degrees of freedom), which
            # is what :meth:`_effective_ndof` uses -- see the note there.
            self.hat_reco_np = None
            try:
                K = self._th2_to_np(unfold.GetProbabilityMatrix("bl_K", "K", False))
                J = self._th2_to_np(unfold.GetDXDY("bl_dxdy", "dxdy", False))
                n_reco = self.mosaic.shape[0]
                # TUnfold returns both matrices as (gen, reco); K needs the
                # transpose to act reco <- gen, J is already gen <- reco.  The
                # shape tests below disambiguate only when n_reco != n_true —
                # for a SQUARE response (pair-split ungroomed) they both fire
                # and wrongly flip J, so guard them with the inequality and
                # apply the known raw orientation directly when square.
                if n_reco == n_true:
                    K = K.T
                else:
                    if K.shape == (n_true, n_reco):
                        K = K.T
                    if J.shape == (n_reco, n_true):
                        J = J.T
                if K.shape == (n_reco, n_true) and J.shape == (n_true, n_reco):
                    hat = K @ J                     # reco -> reco hat matrix H
                    trace = float(np.trace(hat))
                    # Sanity: tr(H) is the effective dof and must land in
                    # (0, ~n_gen] (tau = 0 gives ~n_gen).  A wrong orientation
                    # produces a nonsense trace; fall back to covariance-rank
                    # ndof rather than dividing chi2 by garbage.
                    if 0.0 < trace <= 1.05 * min(n_reco, n_true) + 1.0:
                        self.hat_reco_np = hat
                    else:
                        self.hat_reco_np = None
            except Exception:
                self.hat_reco_np = None

        if systematic == "herwigUp":
            self.cov_uncorr_data = unfold.GetEmatrixInput(
                "cov_uncorr_data",
                "Covariance Matrix from Stat Uncertainties of Input Data",
            )
            _, n_true = self.mosaic.shape
            self.cov_data_herwig_np = np.zeros((n_true, n_true))
            for i in range(1, n_true + 1):
                for j in range(1, n_true + 1):
                    self.cov_data_herwig_np[i - 1, j - 1] = self.cov_uncorr_data.GetBinContent(i, j)


    @staticmethod
    def _th2_to_np(h):
        """Dense (nbinsX, nbinsY) array of a ROOT TH2, excluding over/underflow."""
        nx, ny = h.GetNbinsX(), h.GetNbinsY()
        arr = np.empty((nx, ny))
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                arr[i - 1, j - 1] = h.GetBinContent(i, j)
        return arr


    def _effective_ndof(self, reco_idx):
        """Bottom-line N_dof: effective rank of K J J^T K^T over a reco block.

        The CMS Statistics Committee / TUnfold TWiki prescribes, under
        regularization, ``N_dof = effective rank of K J J^T K^T`` for the
        chi-square test in the unfolded space (the raw covariance rank
        over-counts modes the regularization has suppressed).

        The reco-space hat matrix ``H = K J`` is an *oblique* projection
        (n_reco > n_gen), so its singular values differ from its eigenvalues
        and the participation ratio of ``K J J^T K^T = H H^T`` is inflated by a
        few large singular modes -- it does not count degrees of freedom.  The
        robust, scale-free equivalent is the effective number of degrees of
        freedom ``tr(H)`` (Wahba's effective dof): it equals the rank of
        ``K J J^T K^T`` when the unfolding is unregularized and shrinks
        smoothly as regularization suppresses modes.  ``tr(H)`` over a reco
        sub-block (the summed leverages) is the N_dof carried by that block.

        Returns a float, or ``None`` when the TUnfold matrices are unavailable
        (e.g. the RooUnfold backend), so callers fall back to the covariance
        rank.
        """
        H = getattr(self, "hat_reco_np", None)
        if H is None:
            return None
        idx = np.asarray(list(reco_idx), dtype=int)
        return float(np.trace(H[np.ix_(idx, idx)]))


    @staticmethod
    def _chi2_from_covariance(delta, covariance, *, rcond=1e-12, ndof=None):
        """Return a stable correlated chi-square and the covariance rank.

        A per-pT normalization or an exactly constrained unfolding covariance
        can be singular by construction.  Diagonalizing the symmetric part and
        retaining only resolved positive eigenmodes gives the appropriate
        pseudo-inverse statistic and its corresponding number of degrees of
        freedom.

        ``ndof`` overrides the number of degrees of freedom used for the
        p-value (and reported as ``ndof``).  The CMS Statistics Committee /
        TUnfold TWiki bottom-line test prescribes, for the *unfolded* space
        under regularization, the effective rank of ``K J J^T K^T`` rather
        than the raw covariance rank -- see :meth:`_effective_ndof`.  The
        covariance rank is still reported separately as ``cov_rank``.
        """
        delta = np.asarray(delta, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (delta.size, delta.size):
            raise ValueError(
                "Covariance shape does not match the tested residual vector: "
                f"{covariance.shape} versus {delta.size}."
            )

        symmetric_covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        keep = eigenvalues > rcond * scale
        rank = int(np.count_nonzero(keep))
        if rank == 0:
            raise ValueError("Bottom-line covariance has no resolved positive modes.")

        projected_residual = eigenvectors[:, keep].T @ delta
        chi2_value = float(np.sum(projected_residual**2 / eigenvalues[keep]))
        condition = float(np.max(eigenvalues[keep]) / np.min(eigenvalues[keep]))
        dof = rank if ndof is None else float(ndof)
        return {
            "chi2": chi2_value,
            "ndof": dof,
            "cov_rank": rank,
            "pvalue": float(scipy_chi2.sf(chi2_value, dof)) if dof > 0 else float("nan"),
            "condition": condition,
        }


    def _model_cov_unfolded(self, gidx):
        """Correlated PS/HAD modelling covariance on the unfolded data, in
        the un-normalized gen space, restricted to flat gen indices ``gidx``.

        ARC round-2: the bottom-line test folds the modelling uncertainty into
        the *unfolded data* covariance (the detector-level data carries no
        modelling term, so the smeared side is untouched). The stored
        The stored signed PS/HAD shifts are fractional variations of the
        normalized per-pT shape. Multiplying them by the un-normalized
        ``y_unf`` gives the absolute nuisance vectors (the per-pT normalization
        is a common scale, so shape-fraction == count-fraction to the accuracy
        of this test). Their outer products retain the bin-to-bin correlations.
        Returns ``None`` when no modelling envelope is available.
        """
        if self._uses_enclosing_model_covariance():
            # Convert the normalized density covariance back to count units
            # without dividing by nominal contents, which may be zero.
            scale = np.empty_like(self.y_unf, dtype=float)
            offset = 0
            for i, edges in enumerate(self.gen_edges_by_pt):
                block = slice(offset, offset + len(edges) - 1)
                total = np.sum(np.asarray(self.y_unf)[block][self._shown_gen_mask(i)])
                scale[block] = np.diff(edges) * total
                offset = block.stop
            covariance = sum(self.model_group_covariances.values()) * np.outer(scale, scale)
            return covariance[np.ix_(list(gidx), list(gidx))]
        scope = getattr(
            getattr(self, "spec", None),
            "model_covariance_scope",
            "global_shown",
        )
        ps_frac = getattr(self, "model_ps_shift_flat", None)
        had_frac = getattr(self, "model_had_shift_flat", None)
        if ps_frac is None or had_frac is None:
            return None
        gidx = list(gidx)
        y = np.asarray(self.y_unf, float)[gidx]
        if scope == "per_pt":
            covariance = np.zeros((len(gidx), len(gidx)), dtype=float)
            for ps_full, had_full in zip(
                    self.model_ps_shifts_by_pt_flat,
                    self.model_had_shifts_by_pt_flat):
                ps_abs = np.asarray(ps_full, float)[gidx] * y
                had_abs = np.asarray(had_full, float)[gidx] * y
                covariance += (
                    np.outer(ps_abs, ps_abs) + np.outer(had_abs, had_abs)
                )
            return covariance
        if scope not in ("global_shown", "global_all"):
            raise ValueError(f"Unknown model_covariance_scope: {scope}")
        ps_abs = np.asarray(ps_frac, float)[gidx] * y
        had_abs = np.asarray(had_frac, float)[gidx] * y
        return np.outer(ps_abs, ps_abs) + np.outer(had_abs, had_abs)


    def bottom_line_test(self):
        """Evaluate the bottom-line test for the nominal model.

        The detector-space comparison is between the fake-corrected measured
        data and the matched reconstructed PYTHIA prediction.  The unfolded
        comparison is between the unfolded data and the full PYTHIA truth
        (matched plus misses).  Both use only propagated *data* statistics:
        fake rates, response MC statistics, detector systematics, and any
        model/bias covariance are deliberately excluded here.

        This is the quantitative counterpart to :meth:`plot_bottom_line`.
        For the unregularized production configuration, the expected result is
        ``chi2_unfolded <= chi2_smeared``.  Equality is not required because
        the detector-space fit can contain residual modes that the finite truth
        parameterization cannot represent.
        """
        measured_variances = getattr(self, "corrected_measured_variances", None)
        if measured_variances is None:
            raise RuntimeError(
                "No fake-corrected data sumw2 is available for the bottom-line "
                "test. Re-run with input data histogram variances enabled."
            )
        unfolded_covariance = getattr(self, "cov_data_np", None)
        if unfolded_covariance is None:
            raise RuntimeError(
                "No propagated input-data covariance is available for the "
                "bottom-line test."
            )

        # The response matrix contains matched events only.  The fake
        # correction therefore puts data and model in the same reco space.
        reco_residual = np.asarray(self.y_meas, float) - np.asarray(
            self.mosaic.sum(axis=1), float
        )
        truth_residual = np.asarray(self.y_unf, float) - np.asarray(self.y_true, float)
        smeared = self._chi2_from_covariance(
            reco_residual,
            np.diag(np.asarray(measured_variances, dtype=float)),
        )
        ndof_unf = self._effective_ndof(range(reco_residual.size))
        unfolded_stat = self._chi2_from_covariance(
            truth_residual, unfolded_covariance, ndof=ndof_unf,
        )
        model_cov = self._model_cov_unfolded(range(truth_residual.size))
        if model_cov is not None:
            unfolded = self._chi2_from_covariance(
                truth_residual, unfolded_covariance + model_cov, ndof=ndof_unf,
            )
        else:
            unfolded = unfolded_stat
        return {
            "scope": ("data stat + modelling uncertainty on the unfolded data"
                      if model_cov is not None else "data-stat-only"),
            "model_uncertainty_included": model_cov is not None,
            "regularization": self.regularization,
            "tau": float(self.tau) if self.tau is not None else 0.0,
            "smeared": smeared,
            "unfolded": unfolded,
            "unfolded_stat_only": unfolded_stat,
            "chi2_difference_unfolded_minus_smeared": (
                unfolded["chi2"] - smeared["chi2"]
            ),
            "inequality_holds": bool(unfolded["chi2"] <= smeared["chi2"] + 1e-9),
        }


    def _bl_shown_floors(self):
        """Per-pT rho floor of the SHOWN space (aligned with ``pt_edges``).

        Uses the spec's ``bl_shown_floors_{groomed,ungroomed}`` when set (so the
        chi2 is evaluated only up to the rho value each slice is cut at for
        display), else falls back to the uniform ``xlim_lower`` for the mode.
        """
        floors = (self.spec.bl_shown_floors_groomed if self.groomed
                  else self.spec.bl_shown_floors_ungroomed)
        if floors is not None:
            return list(floors)
        lo = (self.spec.xlim_lower_groomed if self.groomed
              else self.spec.xlim_lower_ungroomed)
        return [lo] * (len(self.pt_edges) - 1)


    def _normalization_window(self):
        """Return the optional transformed-rho normalization interval."""
        window = (self.spec.normalization_window_groomed if self.groomed
                  else self.spec.normalization_window_ungroomed)
        if window is None:
            return None
        if len(window) != 2 or not window[0] < window[1]:
            raise ValueError(
                "normalization window must be a two-element (low, high) interval"
            )
        return float(window[0]), float(window[1])


    def _display_window(self):
        """Return the transformed-rho interval drawn by result panels.

        Preserve the historical coupling when no display-only interval is
        configured: existing specs that set a normalization window continue
        to crop to that same window.
        """
        attribute = (
            "display_window_groomed" if self.groomed
            else "display_window_ungroomed"
        )
        window = getattr(self.spec, attribute, None)
        if window is None:
            return self._normalization_window()
        if len(window) != 2 or not window[0] < window[1]:
            raise ValueError(
                "display window must be a two-element (low, high) interval"
            )
        return float(window[0]), float(window[1])


    @staticmethod
    def _complete_bin_mask(edges, window):
        """Select bins fully contained in ``window``."""
        edges = np.asarray(edges, dtype=float)
        if window is None:
            return np.ones(edges.size - 1, dtype=bool)
        low, high = window
        return ((edges[:-1] >= low - 1e-9)
                & (edges[1:] <= high + 1e-9))


    def _normalization_mask(self, edges, i):
        """Select the bins entering one per-pT normalization denominator."""
        edges = np.asarray(edges, dtype=float)
        window = self._normalization_window()
        if window is not None:
            # Keep complete bins only: a high-rho catch-all beginning at high
            # (or a bin crossing either boundary) cannot leak into the shape.
            return self._complete_bin_mask(edges, window)
        if not getattr(self.spec, "normalize_over_shown", False):
            return np.ones(len(edges) - 1, dtype=bool)
        return edges[:-1] >= self._bl_shown_floors()[i] - 1e-9


    def _windowed_display_slice(self, edges, mask, *, context):
        """Return one contiguous display slice for an explicit rho window."""
        selected = np.flatnonzero(np.asarray(mask, dtype=bool))
        if selected.size == 0:
            raise ValueError(
                f"Display window selects no {context} bins"
            )
        if not np.array_equal(selected, np.arange(selected[0], selected[-1] + 1)):
            raise ValueError(
                f"Display window selected non-contiguous {context} bins"
            )
        return slice(int(selected[0]), int(selected[-1]) + 1)


    def _gen_display_slice(self, i, *, legacy_low_prefix=False):
        """Bins to display for a gen-level panel and their matching edges.

        An explicit display window removes both the low migration buffer and
        high catch-all.  When no display-only window is configured, an
        explicit normalization window retains the historical coupled behavior.
        """
        edges = np.asarray(self.gen_edges_by_pt[i], dtype=float)
        display_window = self._display_window()
        if display_window is not None:
            display_slice = self._windowed_display_slice(
                edges,
                self._complete_bin_mask(edges, display_window),
                context="gen",
            )
        elif legacy_low_prefix:
            display_slice = slice(int(np.argmax(self._shown_gen_mask(i))), None)
        else:
            display_slice = slice(None)
        start = 0 if display_slice.start is None else display_slice.start
        stop = len(edges) - 1 if display_slice.stop is None else display_slice.stop
        return display_slice, edges[start:stop + 1]


    def _reco_display_slice(self, i):
        """Reco counterpart to :meth:`_gen_display_slice` for bottom-line plots."""
        edges = np.asarray(self.reco_edges_by_pt[i], dtype=float)
        display_window = self._display_window()
        if display_window is None:
            return slice(None), edges
        display_slice = self._windowed_display_slice(
            edges,
            self._complete_bin_mask(edges, display_window),
            context="reco",
        )
        return display_slice, edges[display_slice.start:display_slice.stop + 1]


    def _display_xlim(self, i_pt=None):
        """Use an explicit display interval when one is configured."""
        window = self._display_window()
        return window if window is not None else self._observable_xlim(i_pt)


    def _shown_gen_mask(self, i):
        """Boolean mask over the gen bins in the reported normalization space."""
        return self._normalization_mask(self.gen_edges_by_pt[i], i)


    def _shown_reco_mask(self, i):
        """Reco-binning companion of :meth:`_shown_gen_mask`."""
        return self._normalization_mask(self.reco_edges_by_pt[i], i)


    def _shown_norm_total(self, values, i, reco=False):
        """Slice-normalization denominator: sum of ``values`` over shown bins."""
        mask = self._shown_reco_mask(i) if reco else self._shown_gen_mask(i)
        return float(np.asarray(values, dtype=float)[mask].sum())


    def _normalized_slice(self, values, edges, i, *, reco=False):
        """Normalize a pT slice with the same window used by result outputs."""
        values = np.asarray(values, dtype=float)
        widths = np.diff(np.asarray(edges, dtype=float))
        total = self._shown_norm_total(values, i, reco=reco)
        if total <= 0.0:
            return np.zeros_like(values)
        return values / widths / total


    def _slice_normalization_jacobian(self, values, edges, i, *, reco=False):
        """Jacobian of :meth:`_normalized_slice` in one pT slice."""
        values = np.asarray(values, dtype=float)
        widths = np.diff(np.asarray(edges, dtype=float))
        mask = (self._shown_reco_mask(i) if reco else self._shown_gen_mask(i)).astype(float)
        total = float((values * mask).sum())
        if total <= 0.0:
            return np.zeros((values.size, values.size), dtype=float)
        return (
            np.eye(values.size) - np.outer(values, mask) / total
        ) / (widths[:, None] * total)


    @staticmethod
    def _reco_to_gen_rebin_1d(reco_vals, reco_edges, gen_edges):
        """Sum reco-binned values into gen bins (reco edges nest in gen edges).
        Works for counts and for variances (both add)."""
        reco_edges = np.asarray(reco_edges, float)
        gen_edges = np.asarray(gen_edges, float)
        reco_vals = np.asarray(reco_vals, float)
        out = np.zeros(len(gen_edges) - 1)
        for k in range(len(gen_edges) - 1):
            m = ((reco_edges[:-1] >= gen_edges[k] - 1e-9)
                 & (reco_edges[1:] <= gen_edges[k + 1] + 1e-9))
            out[k] = reco_vals[m].sum()
        return out


    def bottom_line_test_by_pt(self, min_edge=None, smeared_binning="reco"):
        """Per-pT-slice bottom-line test, plus a reported-range global row.

        ``smeared_binning="gen"`` rebins the detector level onto the gen binning
        before forming chi2_smeared (the ARC's alternative: with the reco side
        rebinned to gen, K is square and chi2_unfold == chi2_smeared is the
        expectation for an unregularized unfold). Default "reco" keeps the
        native 2x-finer reco binning.

        Same residual / covariance definitions as :meth:`bottom_line_test`,
        sliced per reported pT bin (>= 200 GeV; the 0-200 GeV migration sink is
        excluded). Each entry carries the ``smeared`` and ``unfolded``
        ``_chi2_from_covariance`` dicts. Used by the per-panel annotation and by
        the bottom-line chi2 bar chart. Returns ``(rows, global_row)`` or
        ``([], None)`` when the data-stat inputs are unavailable.

        ``min_edge`` restricts both the gen and reco chi2 sums to bins whose
        LOWER edge is >= the floor. It may be a scalar (uniform floor), a
        per-pT sequence aligned with ``pt_edges``, or the string ``"shown"`` to
        use the displayed bins. With an explicit normalization window, that
        means bins fully contained in both its low and high boundaries.
        """
        use_normalization_window = (
            min_edge == "shown" and self._normalization_window() is not None
        )
        if min_edge == "shown" and not use_normalization_window:
            min_edge = self._bl_shown_floors()
        var_y = getattr(self, "corrected_measured_variances", None)
        cov_x = getattr(self, "cov_data_np", None)
        if var_y is None or cov_x is None:
            return [], None
        var_y = np.asarray(var_y, dtype=float)
        cov_x = np.asarray(cov_x, dtype=float)
        reco_resid = np.asarray(self.y_meas, float) - np.asarray(self.mosaic.sum(axis=1), float)
        truth_resid = np.asarray(self.y_unf, float) - np.asarray(self.y_true, float)

        def offsets(edges_by_pt):
            counts = [len(e) - 1 for e in edges_by_pt]
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
            return starts, counts

        gstart, gcount = offsets(self.gen_edges_by_pt)
        rstart, rcount = offsets(self.reco_edges_by_pt)

        if getattr(self.spec, "bottom_line_scale_mc_per_pt", False):
            # Per-slice shape test: scale the MC side of each residual to the
            # data/unfolded yield over the normalization-window bins of that
            # slice (see the spec knob's comment for the physics rationale).
            mc_reco = np.asarray(self.mosaic.sum(axis=1), float).copy()
            true_scaled = np.asarray(self.y_true, float).copy()
            y_meas_arr = np.asarray(self.y_meas, float)
            y_unf_arr = np.asarray(self.y_unf, float)
            for i in range(len(self.pt_edges) - 1):
                r_slice = slice(rstart[i], rstart[i] + rcount[i])
                g_slice = slice(gstart[i], gstart[i] + gcount[i])
                reco_mask = self._shown_reco_mask(i)
                gen_mask = self._shown_gen_mask(i)
                data_total = float(y_meas_arr[r_slice][reco_mask].sum())
                mc_total = float(mc_reco[r_slice][reco_mask].sum())
                if data_total > 0.0 and mc_total > 0.0:
                    mc_reco[r_slice] *= data_total / mc_total
                unf_total = float(y_unf_arr[g_slice][gen_mask].sum())
                true_total = float(true_scaled[g_slice][gen_mask].sum())
                if unf_total > 0.0 and true_total > 0.0:
                    true_scaled[g_slice] *= unf_total / true_total
            reco_resid = y_meas_arr - mc_reco
            truth_resid = y_unf_arr - true_scaled

        rows, g_gen, g_reco = [], [], []
        g_sm_resid, g_sm_var = [], []   # gen-rebinned smeared accumulators
        for i in self._reported_pt_indices():
            if self.pt_edges[i] < 200:
                continue
            gidx = list(range(gstart[i], gstart[i] + gcount[i]))
            ridx = list(range(rstart[i], rstart[i] + rcount[i]))
            if use_normalization_window:
                gen_mask = self._shown_gen_mask(i)
                reco_mask = self._shown_reco_mask(i)
                gidx = [gstart[i] + j for j in np.flatnonzero(gen_mask)]
                ridx = [rstart[i] + j for j in np.flatnonzero(reco_mask)]
                if not gidx or not ridx:
                    continue
            elif min_edge is not None:
                floor_i = (min_edge[i] if isinstance(min_edge, (list, tuple, np.ndarray))
                           else min_edge)
                gedges = np.asarray(self.gen_edges_by_pt[i], float)
                redges = np.asarray(self.reco_edges_by_pt[i], float)
                gidx = [gstart[i] + j for j in range(gcount[i])
                        if gedges[j] >= floor_i - 1e-9]
                ridx = [rstart[i] + j for j in range(rcount[i])
                        if redges[j] >= floor_i - 1e-9]
                if not gidx or not ridx:
                    continue
            hi = int(self.pt_edges[i + 1]) if i + 1 < len(self.pt_edges) - 1 else None
            cov_unf = cov_x[np.ix_(gidx, gidx)]
            unf_stat = self._chi2_from_covariance(
                truth_resid[gidx], cov_unf, ndof=self._effective_ndof(ridx))
            model_cov = self._model_cov_unfolded(gidx)
            unf = self._chi2_from_covariance(
                truth_resid[gidx], cov_unf + model_cov,
                ndof=self._effective_ndof(ridx),
            ) if model_cov is not None else unf_stat
            if smeared_binning == "gen":
                r_full = slice(rstart[i], rstart[i] + rcount[i])
                resid_g = self._reco_to_gen_rebin_1d(
                    reco_resid[r_full], self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                var_g = self._reco_to_gen_rebin_1d(
                    var_y[r_full], self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                gen_local = [gi - gstart[i] for gi in gidx]
                sm_resid_i = resid_g[gen_local]
                sm_var_i = var_g[gen_local]
                smeared_row = self._chi2_from_covariance(sm_resid_i, np.diag(sm_var_i))
                g_sm_resid.append(sm_resid_i)
                g_sm_var.append(sm_var_i)
            else:
                smeared_row = self._chi2_from_covariance(reco_resid[ridx], np.diag(var_y[ridx]))
            rows.append({
                "i": i, "pt_lo": int(self.pt_edges[i]), "pt_hi": hi,
                "smeared": smeared_row,
                "unfolded": unf,
                "unfolded_stat_only": unf_stat,
            })
            g_gen.extend(gidx)
            g_reco.extend(ridx)
        if not rows:
            return [], None
        g_model_cov = self._model_cov_unfolded(g_gen)
        g_cov = cov_x[np.ix_(g_gen, g_gen)]
        g_unf_stat = self._chi2_from_covariance(
            truth_resid[g_gen], g_cov, ndof=self._effective_ndof(g_reco))
        if smeared_binning == "gen":
            g_sm_resid = np.concatenate(g_sm_resid)
            g_sm_var = np.concatenate(g_sm_var)
            glob_smeared = self._chi2_from_covariance(g_sm_resid, np.diag(g_sm_var))
        else:
            glob_smeared = self._chi2_from_covariance(reco_resid[g_reco], np.diag(var_y[g_reco]))
        glob = {
            "smeared": glob_smeared,
            "unfolded": (self._chi2_from_covariance(
                truth_resid[g_gen], g_cov + g_model_cov,
                ndof=self._effective_ndof(g_reco))
                if g_model_cov is not None else g_unf_stat),
            "unfolded_stat_only": g_unf_stat,
            "model_uncertainty_included": g_model_cov is not None,
        }
        return rows, glob


    @staticmethod
    def _fmt_ndof(ndof):
        """Integer bin-count ndof prints cleanly; effective (float) ndof to .1f."""
        if ndof is None or not np.isfinite(ndof):
            return "-"
        return f"{ndof:.0f}" if abs(ndof - round(ndof)) < 1e-6 else f"{ndof:.1f}"


    def _bottom_line_panel_text(self, slice_row, global_row):
        """Annotation text: the global chi2/ndof (the headline number the ARC
        asked for) plus this slice's chi2/ndof, evaluated over the shown space."""
        def rndf(metric):
            n = metric.get("ndof", float("nan"))
            return metric["chi2"] / n if n and np.isfinite(n) and n > 0 else float("nan")

        with_model = bool((global_row or slice_row or {}).get(
            "model_uncertainty_included", False))
        unc = "stat + model" if with_model else "data stat"
        lines = []
        if global_row is not None:
            gsm, gun = global_row["smeared"], global_row["unfolded"]
            grsm, grun = rndf(gsm), rndf(gun)
            gok = grun <= grsm + 1e-9
            lines += [
                fr"Global $\chi^2/n_\mathrm{{dof}}$ ({unc}, shown range):",
                fr"  smeared $={grsm:.2f}$   unfolded $={grun:.2f}$  "
                + (r"$\checkmark$" if gok else r"$\times$"),
            ]
        if slice_row is not None:
            sm, un = slice_row["smeared"], slice_row["unfolded"]
            rsm, run_ = rndf(sm), rndf(un)
            lines += [
                "",
                fr"This slice: $\chi^2/n_\mathrm{{dof}}$ "
                fr"smear $={rsm:.2f}$, unfold $={run_:.2f}$",
                fr"  ($\chi^2_\mathrm{{unf}}={un['chi2']:.0f}/{self._fmt_ndof(un['ndof'])}$, "
                fr"$\chi^2_\mathrm{{sm}}={sm['chi2']:.0f}/{self._fmt_ndof(sm['ndof'])}$)",
            ]
        return "\n".join(lines)


    def _store_unfold_result(self, systematic, do_jk, jk_target, unfold, h_meas, h_true):
        h_unfold = unfold.GetOutput("unfold")
        h_folded = unfold.GetFoldedOutput("folded")

        y_meas, ye_meas = self._th1_to_arrays(h_meas)
        y_true, ye_true = self._th1_to_arrays(h_true)
        x_folded, _ = self._th1_to_arrays(h_folded)
        y_unf, ye_unf = self._th1_to_arrays(h_unfold)

        if do_jk and systematic == "nominal":
            if jk_target == "matrix":
                self.y_unf_jk_matrix_list.append(y_unf)
            else:
                self.y_unf_jk_input_list.append(y_unf)
            return

        if systematic == "nominal":
            self.y_meas = y_meas
            self.ye_meas = ye_meas
            self.y_unf = y_unf
            self.ye_unf = ye_unf
            self.y_true = y_true
            self.x_folded = x_folded
            self.L = unfold.GetL("Lmatrix", "Lmatrix")
        else:
            self.y_unf_dict[systematic] = y_unf
            self.ye_unf_dict[systematic] = ye_unf


    def _perform_unfold(self, systematic = 'nominal', closure = False, herwig_closure = False, meas_flat = None, do_jk = False, resp_np = None, jk_target = "input", meas_var = None, true_flat_override = None):
        if getattr(self, "method", "tunfold") == "roounfold_bayes":
            return self._perform_unfold_bayes(
                systematic=systematic, closure=closure, herwig_closure=herwig_closure,
                meas_flat=meas_flat, do_jk=do_jk, resp_np=resp_np, jk_target=jk_target,
            )
        uses_default_measurement = meas_flat is None and not closure and not herwig_closure
        uses_stored_matrix = resp_np is None
        if resp_np is None:
            resp_np = self.mosaic_dict[systematic]
        meas_flat = self._select_measured_spectrum(closure, herwig_closure, meas_flat)
        meas_flat = self._apply_background_subtraction(meas_flat, closure, herwig_closure)
        meas_flat = self._apply_fake_correction(meas_flat, systematic, closure, herwig_closure)

        # The truth prior doubles as the regularization bias (ratio-curvature
        # conditions + bookkeeping). A caller unfolding through a reweighted
        # response passes the matching reweighted truth so the bias is consistent.
        if true_flat_override is not None:
            true_flat = np.asarray(true_flat_override, dtype=float)
        else:
            true_flat = self.mosaic.sum(axis = 0) + self.misses_2d
        n_reco, n_true = resp_np.shape
        assert len(meas_flat) == n_reco, "measured spectrum must have n_reco bins"
        truth_root, reco_root = self._build_root_binning()
        h_meas = reco_root.CreateHistogram("hRecoData")
        h_true = truth_root.CreateHistogram("hTruthPrior")
        h_resp = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(truth_root, reco_root, "hResponse")

        if systematic in {"herwigUp", "herwigDown"}:
            misses = self.misses_2d_herwig
        else:
            # Per-systematic misses keep the efficiency consistent with the varied
            # response matrix. Prepared inputs must provide them explicitly --
            # combining a varied response with nominal efficiency is invalid.
            prepared_misses = getattr(self, "misses_2d_dict", None)
            if getattr(self, "_uses_prepared_inputs", False):
                if prepared_misses is None:
                    raise ValueError("Prepared inputs did not define per-systematic misses")
                if systematic not in prepared_misses:
                    raise ValueError(
                        f"Prepared misses are missing systematic {systematic!r}"
                    )
                misses = prepared_misses[systematic]
            else:
                misses = self.misses_2d
        # Proper MC-stat errors (sumw2 from the pkl hists) exist for the
        # stored nominal matrix; JK replica matrices (resp_np override) and
        # systematic variations fall back to the previous behavior.
        resp_var = (
            getattr(self, "mosaic_var_dict", {}).get(systematic)
            if uses_stored_matrix
            else None
        )
        misses_var = (
            getattr(self, "misses_var_dict", {}).get(systematic)
            if uses_stored_matrix
            else None
        )
        if (uses_stored_matrix and getattr(self, "_uses_prepared_inputs", False)
                and (resp_var is None or misses_var is None)):
            raise ValueError(
                f"Prepared response statistics are missing for {systematic!r}"
            )
        self._fill_response_histogram(
            h_resp,
            resp_np,
            misses,
            include_bin_errors=getattr(
                self,
                "response_matrix_stat_available",
                True,
            ),
            resp_var=resp_var,
            misses_var=misses_var,
        )
        measured_variances = None
        corrected_measured_covariance = None
        raw_measured_covariance = (
            getattr(self, "measured_covariance", None)
            if uses_default_measurement else None
        )
        if raw_measured_covariance is not None:
            # The background MC-stat term is diagonal; add it before the
            # systematic-specific fake-survival transform C -> D C D.
            covariance_before_fake = np.array(raw_measured_covariance, copy=True)
            input_variances = np.diag(covariance_before_fake).copy()
            with_background_variances = self._add_background_variance(
                input_variances, closure, herwig_closure
            )
            covariance_before_fake += np.diag(
                with_background_variances - input_variances
            )
            fake_survival = self._fake_survival(systematic)
            corrected_measured_covariance = scale_covariance_by_fake_survival(
                covariance_before_fake, fake_survival
            )
            measured_variances = np.diag(corrected_measured_covariance).copy()
            if systematic == "nominal":
                self.corrected_measured_covariance = np.array(
                    corrected_measured_covariance, copy=True
                )
                self.corrected_measured_variances = np.array(
                    measured_variances, copy=True
                )
            if self.corrected_measured_covariance_dict is not None:
                self.corrected_measured_covariance_dict[systematic] = np.array(
                    corrected_measured_covariance, copy=True
                )
        elif uses_default_measurement and self.measured_variances is not None:
            # Legacy diagonal fallback when an input has no full covariance.
            measured_variances = self._add_background_variance(
                self.measured_variances, closure, herwig_closure
            )
            fake_survival = self._fake_survival(systematic)
            measured_variances = measured_variances * np.square(fake_survival)
            if systematic == "nominal":
                self.corrected_measured_variances = np.array(
                    measured_variances, copy=True
                )
        # Explicit measured variance (e.g. unfolding the HERWIG sample with its
        # own MC-stat in herwig_closure mode, where the default path feeds none).
        # Caller supplies the variance already matched to ``meas_flat``; the fake
        # correction (when applicable) is applied here for consistency with data.
        if meas_var is not None:
            mv = np.asarray(meas_var, dtype=float)
            if not (closure or herwig_closure):
                mv = self._add_background_variance(mv, closure, herwig_closure)
                mv = mv * np.square(1.0 - np.asarray(self.fake_fraction_2d, dtype=float))
            measured_variances = mv
            if systematic == "nominal":
                self.corrected_measured_variances = np.array(mv, copy=True)
        self._fill_root_histogram(h_meas, meas_flat, measured_variances)
        self._fill_root_histogram(h_true, true_flat)
        self.h_resp = h_resp

        # Area constraint: kEConstraintArea (default) keeps one global area
        # constraint; kEConstraintNone disables it (plain TUnfold config).
        e_constraint = (
            ROOT.TUnfold.kEConstraintArea
            if getattr(self.spec, "area_constraint", True)
            else ROOT.TUnfold.kEConstraintNone
        )

        if self.regularization not in {"none", "ratio_curvature", "curvature"}:
            raise ValueError(
                "regularization must be one of 'none', 'ratio_curvature', or "
                f"'curvature', got {self.regularization!r}"
            )
        if self.regularization == "ratio_curvature":
            _declare_open_l()
            unfold = ROOT.TUnfoldDensityOpenL(
                h_resp,
                ROOT.TUnfold.kHistMapOutputHoriz,
                ROOT.TUnfold.kRegModeNone,             # L built by hand below
                e_constraint,
                ROOT.TUnfoldDensity.kDensityModeBinWidth,
                truth_root,
                reco_root,
            )
            # The L matrix uses the *nominal* truth prior for every systematic
            # and jackknife variation: the regularization is part of the
            # measurement definition and must not vary with the response.
            self._add_ratio_curvature_conditions(unfold, true_flat)
        else:
            unfold = ROOT.TUnfoldDensity(
                h_resp,
                ROOT.TUnfold.kHistMapOutputHoriz,          # mapping of TH2 axes
                ROOT.TUnfold.kRegModeNone,
                e_constraint,                              # area constraint (see above)
                ROOT.TUnfoldDensity.kDensityModeBinWidth,  # bin-width aware scaling
                truth_root,                              # output (truth) binning tree
                reco_root,                               # input  (reco)  binning tree
                "",                                      # rows added explicitly below
                ""                                       # no axis-steering fallback
            )
            if self.regularization == "curvature":
                self._add_curvature_regularization(unfold)

        # Feed the full fake-corrected covariance when supplied. TUnfold's
        # fourth SetInput argument is the input covariance TH2; h_meas errors
        # remain the diagonal fallback for legacy/prepared inputs without one.
        if corrected_measured_covariance is not None:
            self._root_input_covariance_hist = self._root_covariance_histogram(
                corrected_measured_covariance,
                f"hInputCovariance_{systematic}",
            )
            status = unfold.SetInput(h_meas, 0.0, 0.0, self._root_input_covariance_hist)
        else:
            status = unfold.SetInput(h_meas)
        if status >= 10000:
            raise RuntimeError("TUnfold input had overflow/underflow – check your hist.")
        if self.regularization == "none":
            unfold.DoUnfold(0.0) # No regularization
        elif self.tau is None:
            # Nominal data unfold (always the first call): scan once, then
            # freeze tau for every systematic / jackknife re-unfold.
            _declare_open_l()
            scan = ROOT.RunUnfoldLcurveScan(unfold, 40)
            self.tau = float(unfold.GetTau())
            x, y = _graph_to_arrays(scan.lcurve)
            self.lcurve_scan = {
                "x": x,
                "y": y,
                "best_x": np.log10(max(unfold.GetChi2A(), 1e-300)),
                "best_y": np.log10(max(unfold.GetChi2L() / max(self.tau, 1e-300) ** 2, 1e-300)),
                "tau": self.tau,
            }
            print(f"L-curve scan: tau = {self.tau:.4g}")
        else:
            unfold.DoUnfold(self.tau)
        # JK replica unfolds also run with systematic == "nominal"; without
        # this guard the last replica (a 90% matrix with no sumw2 errors)
        # silently overwrites the nominal covariances.
        if not do_jk:
            self._store_covariances(unfold, systematic)
        self._store_unfold_result(systematic, do_jk, jk_target, unfold, h_meas, h_true)


    def _perform_unfold_bayes(self, systematic="nominal", closure=False, herwig_closure=False,
                              meas_flat=None, do_jk=False, resp_np=None, jk_target="input"):
        """RooUnfoldBayes (D'Agostini) backend, dispatched from _perform_unfold.

        Mirrors the TUnfold path's input prep (fake-corrected measured spectrum,
        truth-with-misses prior) and result storage, but unfolds with iterative
        Bayes. The jackknife and systematics loops call _perform_unfold
        unchanged, so the same replicas/variations flow through here -- the
        statistical uncertainty stays jackknife-based.
        """
        from unfold.roounfold import bayes_unfold

        uses_default_measurement = meas_flat is None and not closure and not herwig_closure
        if resp_np is None:
            resp_np = self.mosaic_dict[systematic]
        resp_np = np.asarray(resp_np, dtype=float)
        meas_flat = self._select_measured_spectrum(closure, herwig_closure, meas_flat)
        meas_flat = self._apply_fake_correction(meas_flat, systematic, closure, herwig_closure)

        if systematic in {"herwigUp", "herwigDown"}:
            misses = self.misses_2d_herwig
        else:
            misses = getattr(self, "misses_2d_dict", {}).get(systematic, self.misses_2d)
        truth_flat = resp_np.sum(axis=0) + np.asarray(misses, dtype=float)

        want_cov = (not do_jk) and systematic == "nominal" and not herwig_closure
        out = bayes_unfold(resp_np, meas_flat, truth_flat, n_iter=self.n_iter,
                           with_covariance=want_cov, tag=str(systematic))
        if want_cov:
            y_unf, ye_unf, cov = out
        else:
            y_unf, ye_unf = out
            cov = None

        # Jackknife replicas only need the central unfolded vector.
        if do_jk and systematic == "nominal":
            if jk_target == "matrix":
                self.y_unf_jk_matrix_list.append(y_unf)
            else:
                self.y_unf_jk_input_list.append(y_unf)
            return

        if systematic != "nominal":
            self.y_unf_dict[systematic] = y_unf
            if systematic == "herwigUp":
                self.cov_data_herwig_np = np.diag(ye_unf ** 2)
            return

        # Nominal (or closure/herwig-closure): populate the same attributes the
        # TUnfold store path sets. Folded prediction = column-normalized response
        # (including efficiency) applied to the unfolded truth.
        with np.errstate(divide="ignore", invalid="ignore"):
            col = np.divide(resp_np, truth_flat[None, :],
                            out=np.zeros_like(resp_np), where=truth_flat[None, :] != 0)
        self.y_meas = np.asarray(meas_flat, dtype=float)
        self.ye_meas = np.sqrt(np.abs(self.y_meas))
        self.y_unf = y_unf
        self.ye_unf = ye_unf
        self.y_true = truth_flat
        self.x_folded = col @ y_unf
        self.L = None
        # Fake-corrected measured variances (consumed by the dijet artifact
        # writer and normalized-covariance code), mirroring the TUnfold path.
        if uses_default_measurement and self.measured_variances is not None:
            fake_survival = 1.0 - np.asarray(self.fake_fraction_2d, dtype=float)
            self.corrected_measured_variances = (
                self._add_background_variance(
                    np.asarray(self.measured_variances, dtype=float),
                    closure,
                    herwig_closure,
                )
                * np.square(fake_survival)
            )
        if cov is None:
            cov = np.diag(ye_unf ** 2)
        self.cov_np = cov
        self.cov_uncorr_np = cov.copy()
        self.cov_data_np = cov.copy()


    def _th1_to_arrays(self,h):
        nb = h.GetNbinsX()                       # bin numbers
        x  = np.arange(1, nb + 1)
        y  = np.array([h.GetBinContent(int(i)) for i in x])
        ye = np.array([h.GetBinError(int(i))   for i in x])
        return  y, ye


    def _folded_counts_payload(self, i_pt):
        """Return native-reco corrected-count inputs for :meth:`plot_folded`."""
        covariance = getattr(self, "corrected_measured_covariance", None)
        if covariance is None:
            covariance = getattr(self, "measured_covariance", None)
        if covariance is None:
            variances = getattr(self, "corrected_measured_variances", None)
            if variances is None:
                variances = getattr(self, "measured_variances", None)
            covariance = np.diag(np.asarray(variances, dtype=float))
        measured_errors = np.sqrt(
            np.clip(np.diag(np.asarray(covariance, dtype=float)), 0.0, None)
        )
        folded = unflatten_gen_by_pt(self.x_folded, self.reco_edges_by_pt)[i_pt]
        measured = unflatten_gen_by_pt(self.y_meas, self.reco_edges_by_pt)[i_pt]
        errors = unflatten_gen_by_pt(measured_errors, self.reco_edges_by_pt)[i_pt]
        display_slice, edges = self._reco_display_slice(i_pt)
        # Densities, not raw counts: variable-width bins otherwise distort
        # the displayed shape.  The ratio is width-invariant.
        widths = np.diff(np.asarray(edges, dtype=float))
        return {
            "edges": edges,
            "folded": np.asarray(folded, dtype=float)[display_slice] / widths,
            "measured": np.asarray(measured, dtype=float)[display_slice] / widths,
            "measured_error": np.asarray(errors, dtype=float)[display_slice] / widths,
        }


    def _model_reco_signed_shifts(self):
        """Signed per-source model fracs on the normalized reco-level MC.

        Column-scaled Vincia/CR/frag responses projected to reco (see
        ``model_envelope.compute_model_reco_shifts``) plus the FSR PSWeight
        response projections from ``mosaic_dict``. Returns
        ``{source: {i: signed frac array on reco binning}}``, or None when the
        spec does not use the model envelope. Cached after the first call.
        """
        if not getattr(self.spec, "model_envelope", False):
            return None
        cached = getattr(self, "_model_reco_shifts_cache", None)
        if cached is not None:
            return cached
        envelope_source = getattr(
            self.spec, "model_envelope_source", "zjet_offline"
        )
        if envelope_source == "zjet_offline":
            from unfold.model import compute_model_reco_shifts
            print("Computing reco-level model shifts (column-scaled projections)...")
            shifts = compute_model_reco_shifts(self)
        elif envelope_source == "prepared_systematics":
            print("Computing reco-level model shifts from prepared responses...")
            shifts = {}
        else:
            raise ValueError(
                f"unsupported model_envelope_source {envelope_source!r}"
            )
        # Reco-level alternate-model and shower-scale responses. FSR and ISR are
        # the stored PSWeight variations; herwigUp is the HERWIG response mosaic
        # registered by _prepare_herwig_inputs. These are analysis mosaics, so
        # the baseline is the analysis nominal. Used by the bottom-line test,
        # which draws its band from the HERWIG difference and the ISR/FSR
        # envelope rather than from the gen-level PS/HAD envelope.
        nom_proj = unflatten_gen_by_pt(
            self.mosaic_dict["nominal"].sum(axis=1), self.reco_edges_by_pt)
        if envelope_source == "prepared_systematics":
            for source in ("vincia", "cr1", "cr2", "fraghard", "fragsoft"):
                key = f"model_{source}"
                mosaic = self.mosaic_dict.get(key)
                if mosaic is None:
                    raise KeyError(
                        f"prepared model envelope is missing response variation {key!r}"
                    )
                var_proj = unflatten_gen_by_pt(
                    np.asarray(mosaic, float).sum(axis=1), self.reco_edges_by_pt)
                shifts[source] = {}
                for i in range(len(self.reco_edges_by_pt)):
                    nom_total = self._shown_norm_total(nom_proj[i], i, reco=True)
                    alt_total = self._shown_norm_total(var_proj[i], i, reco=True)
                    nom = nom_proj[i] / max(nom_total, 1e-300)
                    alt = var_proj[i] / max(alt_total, 1e-300)
                    shifts[source][i] = np.divide(
                        alt, nom, out=np.ones_like(nom), where=nom != 0
                    ) - 1.0
        for var in ("fsrUp", "fsrDown", "isrUp", "isrDown"):
            mosaic = self.mosaic_dict.get(var)
            if mosaic is None:
                continue
            var_proj = unflatten_gen_by_pt(
                np.asarray(mosaic, float).sum(axis=1), self.reco_edges_by_pt)
            shifts[var] = {}
            for i in range(len(self.reco_edges_by_pt)):
                nom_total = self._shown_norm_total(nom_proj[i], i, reco=True)
                alt_total = self._shown_norm_total(var_proj[i], i, reco=True)
                nom = nom_proj[i] / max(nom_total, 1e-300)
                alt = var_proj[i] / max(alt_total, 1e-300)
                shifts[var][i] = np.divide(
                    alt, nom, out=np.ones_like(nom), where=nom != 0) - 1.0
        self._model_reco_shifts_cache = shifts
        return shifts


    def _pythia_gen_theory_band(self, i):
        """PYTHIA gen-level theory uncertainty (ISR/FSR/q2/PDF) for pt bin ``i``.

        Returns ``(up_frac, down_frac)`` arrays in the same normalized,
        per-bin-width units as ``normalized_results[i]['true']`` (the plotted
        PYTHIA8 shape). Each theory source is symmetrized into an up/down
        deviation of the *normalized* gen shape and the sources are summed in
        quadrature. Detector systematics are excluded -- they do not change the
        gen-level prediction. Returns zeros when no theory variation is
        available (e.g. do_syst=False, or inputs without varied gen columns).
        """
        theory_bases = ("isr", "fsr", "q2", "pdf")
        mosaic_dict = getattr(self, "mosaic_dict", {})
        misses_dict = getattr(self, "misses_2d_dict", {})
        nominal_misses = getattr(self, "misses_2d", None)
        if "nominal" not in mosaic_dict or nominal_misses is None:
            return None, None

        def norm_shape(full_gen_flat):
            pt_binned = unflatten_gen_by_pt(full_gen_flat, self.gen_edges_by_pt)[i]
            bin_widths = np.diff(self.gen_edges_by_pt[i])
            total = self._shown_norm_total(pt_binned, i)
            if total == 0:
                return np.zeros_like(pt_binned, dtype=float)
            return pt_binned / bin_widths / total

        def full_gen(key):
            return mosaic_dict[key].sum(axis=0) + misses_dict.get(key, nominal_misses)

        nominal = norm_shape(full_gen("nominal"))
        up_sq = np.zeros_like(nominal)
        down_sq = np.zeros_like(nominal)
        found = False
        for base in theory_bases:
            deviations = []
            for key in (f"{base}Up", f"{base}Down"):
                if key in mosaic_dict:
                    deviations.append(norm_shape(full_gen(key)) - nominal)
            if not deviations:
                continue
            found = True
            deviations = np.array(deviations)
            up_sq += np.maximum(np.max(deviations, axis=0), 0.0) ** 2
            down_sq += np.maximum(-np.min(deviations, axis=0), 0.0) ** 2

        if not found:
            return None, None
        return np.sqrt(up_sq), np.sqrt(down_sq)


    def _prediction_uncertainty(self, i, kind="pythia"):
        """Total (up, down) uncertainty on a gen prediction curve in pt bin ``i``.

        Returned in the same normalized, per-bin-width units as the plotted
        PYTHIA8 / HERWIG7 curve, so it can be dropped straight into errorbar().
        PYTHIA = gen theory envelope (ISR/FSR/q2/PDF) ⊕ MC stat; HERWIG = MC
        stat only (no theory-weight variations available yet). Returns
        (None, None) when nothing is available.
        """
        if kind == "pythia":
            theory_up, theory_down = self._pythia_gen_theory_band(i)
        else:
            theory_up = theory_down = None

        stat_covariance = self._prediction_stat_covariance(i, kind)
        stat = (None if stat_covariance is None else
                np.sqrt(np.clip(np.diag(stat_covariance), 0.0, None)))

        if theory_up is None and stat is None:
            return None, None
        if theory_up is None:
            return stat, stat
        if stat is None:
            return theory_up, theory_down
        return np.sqrt(theory_up**2 + stat**2), np.sqrt(theory_down**2 + stat**2)


    def _prediction_stat_covariance(self, i, kind="pythia"):
        """Prediction statistics, with a pair-split normalization Jacobian."""
        method = getattr(getattr(self, "spec", None), "prediction_stat_method", "fixed_normalization")
        if method not in {"fixed_normalization", "jacobian"}:
            raise ValueError(f"Unknown prediction_stat_method: {method}")
        values = getattr(self, f"{kind}_gen_val_flat", None)
        variances = getattr(self, f"{kind}_gen_var_flat", None)
        is_closure = getattr(self, "closure", False) or getattr(self, "herwig_closure", False)
        if method == "jacobian" and kind == "pythia" and values is None and not is_closure:
            # Prepared pair-split data comparisons carry inclusive GEN sumw2.
            # Do not add an independent truth-stat term to same-MC closure.
            values = (self.gen_mc_flat_dict or {}).get("nominal")
            variances = (self.gen_mc_var_dict or {}).get("nominal")
        if values is None or variances is None:
            return None
        counts = unflatten_gen_by_pt(np.asarray(values, float), self.gen_edges_by_pt)[i]
        variance = unflatten_gen_by_pt(np.asarray(variances, float), self.gen_edges_by_pt)[i]
        widths = np.diff(self.gen_edges_by_pt[i])
        total = self._shown_norm_total(counts, i)
        if total <= 0:
            return None
        if method == "jacobian":
            from unfold.model import normalized_prediction_covariance
            return normalized_prediction_covariance(
                counts, np.diag(np.clip(variance, 0.0, None)), widths,
                self._shown_gen_mask(i),
            )
        # Preserve the legacy nonpositive-bin convention for frozen Z+jet.
        shape = counts / widths / total
        relative = np.divide(np.sqrt(np.clip(variance, 0.0, None)), counts,
                             out=np.zeros_like(shape), where=counts > 0)
        return np.diag((shape * relative) ** 2)


    def _prediction_chi2_covariance(self, i, kind="pythia"):
        """Slice-local covariance of a gen prediction for the quoted chi2.

        PYTHIA: coherent symmetrized ISR/FSR/q2/PDF shift vectors of the
        normalized gen shape (rank-1 each; every variation is separately
        normalized, so the sum-constraint null space is preserved) plus the
        MC-stat covariance. Pair-split propagates normalization; the legacy
        default uses a fixed denominator. HERWIG: MC-stat only -- no theory weights
        are available, an asymmetry the figure caption must state. Returns
        None when nothing is available (the chi2 then uses the measurement
        covariance alone, as for closure runs).
        """
        n_bins = len(self.gen_edges_by_pt[i]) - 1
        covariance = None

        if kind == "pythia":
            theory_bases = ("isr", "fsr", "q2", "pdf")
            mosaic_dict = getattr(self, "mosaic_dict", {})
            misses_dict = getattr(self, "misses_2d_dict", {})
            nominal_misses = getattr(self, "misses_2d", None)
            if "nominal" in mosaic_dict and nominal_misses is not None:
                def norm_shape(key):
                    flat = mosaic_dict[key].sum(axis=0) + misses_dict.get(
                        key, nominal_misses)
                    pt_binned = unflatten_gen_by_pt(flat, self.gen_edges_by_pt)[i]
                    widths = np.diff(self.gen_edges_by_pt[i])
                    total = self._shown_norm_total(pt_binned, i)
                    if total == 0:
                        return np.zeros_like(pt_binned, dtype=float)
                    return pt_binned / widths / total

                nominal = norm_shape("nominal")
                for base in theory_bases:
                    up_key, down_key = f"{base}Up", f"{base}Down"
                    have_up = up_key in mosaic_dict
                    have_down = down_key in mosaic_dict
                    if not (have_up or have_down):
                        continue
                    if have_up and have_down:
                        shift = 0.5 * (norm_shape(up_key) - norm_shape(down_key))
                    else:
                        shift = norm_shape(up_key if have_up else down_key) - nominal
                    if covariance is None:
                        covariance = np.zeros((n_bins, n_bins))
                    covariance += np.outer(shift, shift)

        stat_covariance = self._prediction_stat_covariance(i, kind)
        if stat_covariance is not None:
            covariance = stat_covariance if covariance is None else covariance + stat_covariance

        return covariance


    def _ensure_herwig_bias_inputs(self):
        """Build the HERWIG reco spectrum needed for the bias test, on demand.

        ``mosaic_herwig_2d`` (the HERWIG matched-reco spectrum, the measured
        input of the bias test) is only assembled by the loader when a herwig
        systematic is requested; the bias test needs it regardless.
        """
        if self.mosaic_herwig_2d is not None:
            return
        from unfold.zjet_inputs import herwig_pieces

        self.h2d_herwig, self.fakes_2d_herwig, self.misses_2d_herwig, self.mosaic_gen_herwig = herwig_pieces(
            self.herwig_4d, self.herwig_4d_gen, self.fakes_herwig, self.misses_herwig, self.spec, self.bins,
        )
        self.mosaic_herwig_2d = merge_mass_flat(self.h2d_herwig, self.edges, self.reco_edges_by_pt)

    def _unfold_herwig_through_pythia(self):
        """Unfold the HERWIG reco spectrum through the PYTHIA response matrix.

        Returns ``(y_unf, ye_unf)``: the flattened unfolded result and TUnfold's
        propagated per-bin uncertainty on it. Runs a one-off nominal unfold in
        ``herwig_closure`` mode (measured = HERWIG reco, response = nominal
        PYTHIA, prior = PYTHIA truth) and restores the nominal data result so
        the rest of ``run_all_plots`` is unaffected. tau is already frozen from
        the nominal data unfold, so no re-scan happens here.
        """
        snapshot_attrs = (
            "y_meas", "ye_meas", "y_unf", "ye_unf", "y_true", "x_folded", "L",
            "h_resp", "cov", "cov_uncorr", "cov_uncorr_data", "cov_total",
            "cov_np", "cov_uncorr_np", "cov_data_np",
        )
        sentinel = object()
        saved = {name: getattr(self, name, sentinel) for name in snapshot_attrs}
        try:
            self._perform_unfold(systematic="nominal", herwig_closure=True)
            return np.array(self.y_unf, copy=True), np.array(self.ye_unf, copy=True)
        finally:
            for name, value in saved.items():
                if value is sentinel:
                    if hasattr(self, name):
                        delattr(self, name)
                else:
                    setattr(self, name, value)


    def _normalize_result(self):
        print("Normalizing results...")
        self.normalized_results = []
        gen_mass_bin_edges_by_pt = self.gen_edges_by_pt
        reco_mass_bin_edges_by_pt = self.reco_edges_by_pt

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, gen_mass_bin_edges_by_pt)
        #print("Unfolded pt binned:", unfolded_pt_binned)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, reco_mass_bin_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), reco_mass_bin_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, gen_mass_bin_edges_by_pt)
        error_pt_binned = unflatten_gen_by_pt(self.ye_unf, gen_mass_bin_edges_by_pt)
        for i in range(len(self.pt_edges)-1):
            bin_widths = np.diff(gen_mass_bin_edges_by_pt[i])
            bin_widths_reco = np.diff(reco_mass_bin_edges_by_pt[i])
            # Denominators run over the SHOWN bins of the slice (full slice
            # unless spec.normalize_over_shown): the published spectra
            # integrate to 1 over the reported fiducial window, so hidden
            # buffer / catch-all bins do not enter the normalization.
            result = {
                "true": true_pt_binned[i]/bin_widths/self._shown_norm_total(true_pt_binned[i], i),
                "unfolded": unfolded_pt_binned[i]/bin_widths/self._shown_norm_total(unfolded_pt_binned[i], i),
                "unfolded_err": error_pt_binned[i]/bin_widths/self._shown_norm_total(unfolded_pt_binned[i], i),
                "measured": measured_pt_binned[i]/bin_widths_reco/self._shown_norm_total(measured_pt_binned[i], i, reco=True),
                "reco_mc": reco_mc_pt_binned[i]/bin_widths_reco/self._shown_norm_total(reco_mc_pt_binned[i], i, reco=True),
                "pt_bin": (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf')),
                "mgen_edges": self.gen_edges_by_pt[i]
            }
            self.normalized_results.append(result)
        
        # Storing normalized results for systematics
        self.normalized_systematics = []
        # Prepare normalized_systematics as a list of dicts, one per pt bin

        for i in range(len(self.pt_edges)-1):
            pt_bin = (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf'))
            unfolded = {}
            bin_widths = np.diff(gen_mass_bin_edges_by_pt[i])
            bin_widths_reco = np.diff(reco_mass_bin_edges_by_pt[i])
            for syst in self.systematics:
                if syst == 'nominal':
                    continue
                unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf_dict[syst], gen_mass_bin_edges_by_pt)
                unfolded[syst] = unfolded_pt_binned[i]/bin_widths/self._shown_norm_total(unfolded_pt_binned[i], i)
            self.normalized_systematics.append({
            "pt_bin": pt_bin,
            "unfolded": unfolded # Taking absolute values to avoid negative bins
            })

        self._compute_2d_normalized_result()


    def _compute_2d_normalized_result(self):
        """2D-normalized unfolded result: normalize over the full (m, pT) plane.

        Unlike ``normalized_results`` (each pT slice divided by its own sum so
        every slice integrates to 1), this divides the whole double-differential
        spectrum by a single global integral while keeping the mass-bin density
        (/ bin width). The result is 1/sigma * d^2sigma/(dm dpT), integrating to
        1 over the full (m, pT) plane, so the relative pT-to-pT normalization is
        preserved rather than discarded.

        Stores both the flat (unrolled) arrays and a per-pT-slice list mirroring
        ``normalized_results``, plus a copy of the absolute unfolded output.
        """
        widths_flat = np.concatenate(
            [np.diff(np.asarray(edges, dtype=float)) for edges in self.gen_edges_by_pt]
        )
        unf = np.asarray(self.y_unf, dtype=float)
        true = np.asarray(self.y_true, dtype=float)
        unf_err = np.asarray(self.ye_unf, dtype=float)

        total_unf = unf.sum()
        total_true = true.sum()

        # Preserved copy of the absolute (un-normalized) 2D unfolded output.
        self.unfolded_abs_flat = np.array(unf, copy=True)
        self.unfolded_abs_err_flat = np.array(unf_err, copy=True)

        self.unfolded_2dnorm_flat = unf / widths_flat / total_unf
        self.unfolded_2dnorm_err_flat = unf_err / widths_flat / total_unf
        jackknife_covariance = getattr(self, "jackknife_global_normalized_covariance", None)
        if jackknife_covariance is not None:
            self.unfolded_2dnorm_err_flat = np.sqrt(np.clip(np.diag(jackknife_covariance), 0, None))
        self.true_2dnorm_flat = true / widths_flat / total_true

        unf_pt = unflatten_gen_by_pt(self.unfolded_2dnorm_flat, self.gen_edges_by_pt)
        err_pt = unflatten_gen_by_pt(self.unfolded_2dnorm_err_flat, self.gen_edges_by_pt)
        true_pt = unflatten_gen_by_pt(self.true_2dnorm_flat, self.gen_edges_by_pt)

        self.normalized_2d = []
        for i in range(len(self.pt_edges) - 1):
            self.normalized_2d.append({
                "unfolded": unf_pt[i],
                "unfolded_err": err_pt[i],
                "true": true_pt[i],
                "pt_bin": (
                    self.pt_edges[i],
                    self.pt_edges[i + 1] if i + 1 < len(self.pt_edges) - 1 else float("inf"),
                ),
                "mgen_edges": self.gen_edges_by_pt[i],
            })


    def save_2d_unfolded(self):
        """Pickle the 2D unfolded output (absolute + 2D-normalized) as hists.

        Companion to ``save_normalized_covariance`` (which stores the per-pT
        normalized result). Each entry is a ``hist.Hist`` over (pt, obs) (or a
        per-pT list when the binning is ragged, e.g. mass):
          - "unfolded_abs": absolute unfolded output (Weight, value + TUnfold
            stat variance), in counts.
          - "unfolded_2dnorm": globally (2D) normalized unfolded spectrum
            (Weight, value + stat variance), summing to 1 over the whole plane.
          - "true_2dnorm": PYTHIA gen truth in the same 2D-normalized units.
          - "layout": "2d" or "per_pt"; "pt_edges".

        The 2D-normalized entries are sum-normalized (not divided by bin width);
        recover the density with hist/mplhep ``binwnorm`` at plot time.
        """
        suffix = "groomed" if self.groomed else "ungroomed"
        # Undo the per-bin-width division so the stored 2D-normalized spectrum is
        # sum-normalized (sums to 1 over the plane); apply binwnorm for density.
        widths_flat = np.concatenate(
            [np.diff(np.asarray(e, dtype=float)) for e in self.gen_edges_by_pt]
        )
        summary = {
            "unfolded_abs": self._hist_from_flat(
                self.unfolded_abs_flat, self.unfolded_abs_err_flat
            ),
            "unfolded_2dnorm": self._hist_from_flat(
                self.unfolded_2dnorm_flat * widths_flat,
                self.unfolded_2dnorm_err_flat * widths_flat,
            ),
            "true_2dnorm": self._hist_from_flat(self.true_2dnorm_flat * widths_flat),
            "layout": "2d" if self._slices_share_binning() else "per_pt",
            "pt_edges": np.asarray(self.pt_edges, dtype=float),
        }
        save_path = self._relocate_output(f"unfolded_2d_{suffix}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as handle:
            pkl.dump(summary, handle)
        print(f"Saved 2D unfolded output (hist) to {save_path}")


    def _ptnorm_flat(self, val_flat, err_flat=None):
        """Per-pT-slice normalize a flat gen vector: v / bin_width / slice_sum.

        Matches the convention used for ``normalized_results`` and the MC-stat
        bands (relative error preserved: err / width / slice_sum). Returns the
        normalized values, or ``(values, errors)`` when ``err_flat`` is given.
        """
        val_flat = np.asarray(val_flat, dtype=float)
        out_val = np.zeros_like(val_flat)
        out_err = None if err_flat is None else np.zeros_like(val_flat)
        offset = 0
        for i, edges in enumerate(self.gen_edges_by_pt):
            nbins = len(edges) - 1
            sl = slice(offset, offset + nbins)
            widths = np.diff(np.asarray(edges, dtype=float))
            mask = self._shown_gen_mask(i)
            slice_sum = val_flat[sl][mask].sum()
            if slice_sum != 0:
                out_val[sl] = val_flat[sl] / widths / slice_sum
                if err_flat is not None:
                    out_err[sl] = np.asarray(err_flat[sl], dtype=float) / widths / slice_sum
            offset += nbins
        return (out_val, out_err) if err_flat is not None else out_val


    def _slices_share_binning(self):
        """True when every pT slice has identical gen edges (e.g. rho)."""
        e0 = np.asarray(self.gen_edges_by_pt[0], dtype=float)
        return all(
            np.array_equal(e0, np.asarray(e, dtype=float)) for e in self.gen_edges_by_pt
        )


    def _obs_axis(self, edges):
        return hist.axis.Variable(
            np.asarray(edges, dtype=float), name="obs",
            label=self._observable_short_label(),
            underflow=False, overflow=False,
        )


    def _pt_axis(self):
        return hist.axis.Variable(
            np.asarray(self.pt_edges, dtype=float), name="pt",
            label=r"jet $p_T$ [GeV]", underflow=False, overflow=False,
        )


    def _hist_from_flat(self, val_flat, err_flat=None, extra_axes=()):
        """Build hist(s) from a flat (unrolled) gen vector.

        Returns a single 2D ``Hist(pt, obs[, *extra_axes])`` when all pT slices
        share binning (rho), else a list of per-pT 1D ``Hist(obs[, *extra_axes])``
        (mass, whose binning differs per slice). With ``err_flat`` the storage is
        Weight (value + variance = err**2); otherwise Double. ``extra_axes`` are
        appended after the obs axis and ``val_flat``/``err_flat`` must then be
        shaped (n_bins, *extra_shape) per slice.
        """
        val_flat = np.asarray(val_flat, dtype=float)
        err_flat = None if err_flat is None else np.asarray(err_flat, dtype=float)

        def _fill(axes, val, err):
            if err is None:
                h = hist.Hist(*axes, storage=hist.storage.Double())
                h.view()[...] = val
            else:
                h = hist.Hist(*axes, storage=hist.storage.Weight())
                view = h.view()
                view["value"] = val
                view["variance"] = err ** 2
            return h

        if self._slices_share_binning():
            n_pt = len(self.pt_edges) - 1
            n_obs = len(self.gen_edges_by_pt[0]) - 1
            shape = (n_pt, n_obs) + val_flat.shape[1:]
            axes = (self._pt_axis(), self._obs_axis(self.gen_edges_by_pt[0]), *extra_axes)
            return _fill(axes, val_flat.reshape(shape),
                         None if err_flat is None else err_flat.reshape(shape))

        hists = []
        offset = 0
        for edges in self.gen_edges_by_pt:
            nbins = len(edges) - 1
            sl = slice(offset, offset + nbins)
            axes = (self._obs_axis(edges), *extra_axes)
            hists.append(_fill(axes, val_flat[sl],
                               None if err_flat is None else err_flat[sl]))
            offset += nbins
        return hists


    def save_2d_uncertainty_summary(self):
        """Pickle a per-bin (pT x obs) uncertainty + prediction summary as hists.

        Each entry is a ``hist.Hist`` over (pt, obs) when the gen binning is the
        same in every pT slice (rho), or a list of per-pT 1D ``hist.Hist`` when
        it differs per slice (mass). Requires a full ``do_syst=True`` run for the
        systematic content (otherwise the syst hists are zero).

        Values are sum-normalized per pT slice (each slice integrates to 1, i.e.
        .project("pt") == 1), NOT divided by bin width; recover the density
        (dN/dx) at plot time with hist/mplhep ``binwnorm``. Raw spectra are
        absolute counts.

        Dict keys:
          - "unfolded": Weight hist of the per-pT-normalized unfolded result
            (the usual analysis output); its variance is the per-bin stat.
          - "unfolded_syst", "unfolded_syst_up"/"_down": pure (stat-excluded)
            systematic, symmetric and asymmetric. "unfolded_total_up"/"_down":
            stat (+) syst in quadrature.
          - "syst_breakdown": hist with extra (source, direction) StrCategory
            axes giving the per-source systematic (JES, JER, JMS, ..., Model
            Uncertainty); quadrature over sources reproduces unfolded_syst_*.
            "syst_sources" lists the names.
          - "pythia_gen_raw", "pythia_gen_ptnorm": PYTHIA gen prediction (Weight,
            value + MC-stat variance), absolute and per-pT-normalized.
          - "herwig_gen_raw", "herwig_gen_ptnorm": same for HERWIG; the raw
            HERWIG is rescaled per pT slice to the unfolded-data yield (its
            stored normalization is unreliable), "herwig_pt_scale" holds it.
          - "layout": "2d" or "per_pt"; "pt_edges".
        """
        suffix = "groomed" if self.groomed else "ungroomed"

        # ---- unfolded (per-pT normalized) + stat ----
        unfolded = np.concatenate(
            [np.asarray(r["unfolded"], dtype=float) for r in self.normalized_results]
        )
        stat = np.concatenate(
            [np.asarray(r["stat_unc"], dtype=float) for r in self.normalized_results]
        )

        # ---- systematic breakdown by analysis summary source (stat excluded) ----
        # Each summary source's up/down is the quadrature sum of its individual
        # nuisances per side, matching _compute_total_systematic so that the
        # quadrature over sources reproduces the analysis up/down totals.
        nominal_by_pt = [np.asarray(r["unfolded"], dtype=float) for r in self.normalized_results]
        sources = []
        for syst_name in self.systematics:
            if syst_name == "nominal":
                continue
            src = self._get_systematic_summary_name(syst_name, grouped=False)
            if src not in sources:
                sources.append(src)

        up_sq = {src: [] for src in sources}    # source -> per-pT list of sq-sums
        down_sq = {src: [] for src in sources}
        for i, nominal in enumerate(nominal_by_pt):
            varied = self.normalized_systematics[i]["unfolded"]
            up_acc = {src: np.zeros_like(nominal) for src in sources}
            down_acc = {src: np.zeros_like(nominal) for src in sources}
            for syst_name, arr in varied.items():
                if syst_name == "nominal":
                    continue
                src = self._get_systematic_summary_name(syst_name, grouped=False)
                _, variation = self._split_systematic_variation(syst_name)
                diff_sq = (np.asarray(arr, dtype=float) - nominal) ** 2
                if variation == "Down":
                    down_acc[src] += diff_sq
                else:  # "Up" or unpaired
                    up_acc[src] += diff_sq
            for src in sources:
                up_sq[src].append(up_acc[src])
                down_sq[src].append(down_acc[src])

        syst_per_source_up = {src: np.sqrt(np.concatenate(up_sq[src])) for src in sources}
        syst_per_source_down = {src: np.sqrt(np.concatenate(down_sq[src])) for src in sources}

        n_flat = len(unfolded)
        syst_up = np.sqrt(np.sum([syst_per_source_up[s] ** 2 for s in sources], axis=0)
                          if sources else np.zeros(n_flat))
        syst_down = np.sqrt(np.sum([syst_per_source_down[s] ** 2 for s in sources], axis=0)
                            if sources else np.zeros(n_flat))
        syst_sym = 0.5 * (syst_up + syst_down)
        total_up = np.sqrt(stat ** 2 + syst_up ** 2)
        total_down = np.sqrt(stat ** 2 + syst_down ** 2)

        # ---- generator predictions: raw + per-pT-normalized, with MC stat ----
        def _gen(val_flat, var_flat):
            val = np.asarray(val_flat, dtype=float)
            err = (np.sqrt(np.clip(np.asarray(var_flat, float), 0.0, None))
                   if var_flat is not None else np.zeros_like(val))
            nval, nerr = self._ptnorm_flat(val, err)
            return val, err, nval, nerr

        # PYTHIA/HERWIG gen predictions exist only on the zjet spec path; the
        # dijet/trijet prepared inputs carry none, so emit None for those.
        has_gen_pred = self.pythia_gen_val_flat is not None and self.herwig_gen_val_flat is not None
        if has_gen_pred:
            py_raw, py_raw_err, py_norm, py_norm_err = _gen(
                self.pythia_gen_val_flat, getattr(self, "pythia_gen_var_flat", None)
            )

            # HERWIG's stored gen normalization is unreliable, so rescale each pT
            # slice of the raw HERWIG prediction to the absolute unfolded-data
            # yield in that slice. The per-pT-normalized HERWIG is unaffected (a
            # constant per-slice factor cancels in the normalization).
            hw_scale_flat = np.ones_like(np.asarray(self.herwig_gen_val_flat, dtype=float))
            data_abs = np.asarray(self.y_unf, dtype=float)
            herwig_abs = np.asarray(self.herwig_gen_val_flat, dtype=float)
            offset = 0
            for edges in self.gen_edges_by_pt:
                nbins = len(edges) - 1
                sl = slice(offset, offset + nbins)
                hw_sum = herwig_abs[sl].sum()
                if hw_sum != 0:
                    hw_scale_flat[sl] = data_abs[sl].sum() / hw_sum
                offset += nbins
            hw_var_scaled = (
                np.asarray(self.herwig_gen_var_flat, dtype=float) * hw_scale_flat ** 2
                if getattr(self, "herwig_gen_var_flat", None) is not None
                else None
            )
            hw_raw, hw_raw_err, hw_norm, hw_norm_err = _gen(
                herwig_abs * hw_scale_flat, hw_var_scaled
            )
        else:
            py_raw = py_raw_err = py_norm = py_norm_err = None
            hw_raw = hw_raw_err = hw_norm = hw_norm_err = None
            hw_scale_flat = None

        # Store sum-normalized (per-bin) values, NOT densities: each pT slice
        # integrates to 1 over its bins, so .project("pt") == 1. Recover the
        # density (dN/dx) at plot time with hist/mplhep binwnorm. The normalized
        # quantities above carry a 1/bin_width from _normalize_result/_ptnorm_flat,
        # so multiply it back out here. Raw/absolute spectra are left untouched.
        widths_flat = np.concatenate(
            [np.diff(np.asarray(e, dtype=float)) for e in self.gen_edges_by_pt]
        )
        unfolded = unfolded * widths_flat
        stat = stat * widths_flat
        syst_sym = syst_sym * widths_flat
        syst_up = syst_up * widths_flat
        syst_down = syst_down * widths_flat
        total_up = total_up * widths_flat
        total_down = total_down * widths_flat
        for source in sources:
            syst_per_source_up[source] = syst_per_source_up[source] * widths_flat
            syst_per_source_down[source] = syst_per_source_down[source] * widths_flat
        if has_gen_pred:
            py_norm = py_norm * widths_flat
            py_norm_err = py_norm_err * widths_flat
            hw_norm = hw_norm * widths_flat
            hw_norm_err = hw_norm_err * widths_flat

        # ---- build hist objects (2D when uniform, per-pT list when ragged) ----
        summary = {
            "unfolded": self._hist_from_flat(unfolded, stat),
            "unfolded_syst": self._hist_from_flat(syst_sym),
            "unfolded_syst_up": self._hist_from_flat(syst_up),
            "unfolded_syst_down": self._hist_from_flat(syst_down),
            "unfolded_total_up": self._hist_from_flat(total_up),
            "unfolded_total_down": self._hist_from_flat(total_down),
            "pythia_gen_raw": self._hist_from_flat(py_raw, py_raw_err) if has_gen_pred else None,
            "pythia_gen_ptnorm": self._hist_from_flat(py_norm, py_norm_err) if has_gen_pred else None,
            "herwig_gen_raw": self._hist_from_flat(hw_raw, hw_raw_err) if has_gen_pred else None,
            "herwig_gen_ptnorm": self._hist_from_flat(hw_norm, hw_norm_err) if has_gen_pred else None,
            "herwig_pt_scale": self._hist_from_flat(hw_scale_flat) if has_gen_pred else None,
            "syst_sources": list(sources),
            "layout": "2d" if self._slices_share_binning() else "per_pt",
            "pt_edges": np.asarray(self.pt_edges, dtype=float),
        }

        # systematic breakdown with (source, direction) category axes
        if sources:
            syst_stack = np.zeros((n_flat, len(sources), 2))
            for j, source in enumerate(sources):
                syst_stack[:, j, 0] = syst_per_source_up[source]
                syst_stack[:, j, 1] = syst_per_source_down[source]
            extra = (
                hist.axis.StrCategory(sources, name="source"),
                hist.axis.StrCategory(["up", "down"], name="direction"),
            )
            summary["syst_breakdown"] = self._hist_from_flat(syst_stack, extra_axes=extra)
        else:
            summary["syst_breakdown"] = None

        save_path = self._relocate_output(f"uncertainty_summary_2d_{suffix}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as handle:
            pkl.dump(summary, handle)
        print(f"Saved 2D uncertainty summary (hist) to {save_path}")


    def _jackknife_convergence_fractions(self):
        """Per-bin fractional jackknife stat vs number of replicas.

        Recomputes the grouped-jackknife SE (the same estimator as
        _compute_stat_unc at the full count) of the unfolded result over the
        first n input-data and response-matrix jackknife replicas, for
        n = 2..N. The per-count scale on the sample std (ddof=1) is (n-1)/sqrt(n),
        equal to sqrt(n-1) applied to the population std used in _compute_stat_unc
        (9/sqrt(10) ~ 2.85 at n=10). Returns ``(ns, input_frac, matrix_frac,
        total_frac)`` with the fraction arrays shaped (len(ns), n_gen_bins); or
        None if no replicas.
        """
        input_reps = np.asarray(self.y_unf_jk_input_list, dtype=float)
        matrix_reps = np.asarray(self.y_unf_jk_matrix_list, dtype=float)
        if input_reps.ndim != 2 or matrix_reps.ndim != 2:
            return None
        n_rep = min(len(input_reps), len(matrix_reps))
        if n_rep < 2:
            return None
        ns = np.arange(2, n_rep + 1)
        nominal = np.abs(np.asarray(self.y_unf, dtype=float))

        def fracs(reps):
            rows = []
            for n in ns:
                jk_scale = (n - 1) / np.sqrt(n)
                std = jk_scale * np.std(reps[:n], axis=0, ddof=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rows.append(np.where(nominal > 0, std / nominal, 0.0))
            return np.asarray(rows)

        input_frac = fracs(input_reps)
        matrix_frac = fracs(matrix_reps)
        total_frac = np.sqrt(input_frac**2 + matrix_frac**2)
        return ns, input_frac, matrix_frac, total_frac


    def _normalization_jacobian(self):
        """Jacobian of the per-pT-slice normalization y_i = x_i / (w_i * S_k).

        x is the absolute unfolded spectrum, w_i the gen bin width, and
        S_k the sum of x over the pT slice containing bin i, so
        dy_i/dx_j = (delta_ij - x_i / S_k) / (w_i * S_k) within a slice and
        zero across slices (each slice is normalized by its own sum).
        """
        n = len(self.y_unf)
        jacobian = np.zeros((n, n))
        offset = 0
        for k, edges in enumerate(self.gen_edges_by_pt):
            nbins = len(edges) - 1
            block = slice(offset, offset + nbins)
            x = np.asarray(self.y_unf[block], dtype=float)
            widths = np.diff(edges)
            # S_k sums shown bins only (all bins unless normalize_over_shown):
            # dy_i/dx_j picks up the -x_i/S term only for j in the denominator.
            mask = self._shown_gen_mask(k).astype(float)
            total = float((x * mask).sum())
            if total != 0:
                jacobian[block, block] = (
                    np.eye(nbins) - np.outer(x, mask) / total
                ) / (widths[:, None] * total)
            offset += nbins
        return jacobian


    def _absolute_stat_covariances(self):
        """Covariances of the absolute unfolded spectrum: (input, matrix).

        The selected method supplies these arrays. Legacy Z+jet keeps its
        original analytic covariance convention.
        """
        return (
            np.array(self.cov_data_np, copy=True),
            np.array(self.cov_uncorr_np, copy=True),
        )


    def _compute_normalized_stat_covariance(self):
        """Propagate the absolute stat covariances through the normalization.

        Stores covariances of the normalized, per-bin-width result (the same
        units as ``normalized_results[i]['unfolded']``). The Jacobian is block
        diagonal per pT slice, but cross-slice correlations of the absolute
        covariance survive in the off-diagonal blocks. Within each slice the
        sum constraint makes the covariance singular (one zero eigenvalue) and
        introduces negative correlations; both are expected for a normalized
        measurement.
        """
        replicas = getattr(self, "jackknife_normalized_covariances", None)
        if replicas is not None:
            self.norm_cov_input, self.norm_cov_matrix = (c.copy() for c in replicas)
        else:
            jacobian = self._normalization_jacobian()
            cov_input_abs, cov_matrix_abs = self._absolute_stat_covariances()
            self.norm_cov_input = jacobian @ cov_input_abs @ jacobian.T
            self.norm_cov_matrix = jacobian @ cov_matrix_abs @ jacobian.T
        self.norm_cov_stat = self.norm_cov_input + self.norm_cov_matrix
        if replicas is not None:
            errors = unflatten_gen_by_pt(
                np.sqrt(np.clip(np.diag(self.norm_cov_stat), 0.0, None)), self.gen_edges_by_pt
            )
            for result, error in zip(self.normalized_results, errors):
                result["unfolded_err"] = error


    def get_systematic_covariance(self):
        """Systematic covariance of the normalized result.

        Each source contributes the outer product of its normalized shift,
        symmetrized as (up - down)/2 when both variations exist.  With the
        two-leg model prescription enabled, raw HERWIG/ISR/FSR and
        ``model_*`` inputs are superseded by the two group covariances.
        These use selected coherent vectors in legacy mode, or enclosing
        template ellipsoids when explicitly enabled by the observable spec.
        """
        nominal_flat = np.concatenate(
            [np.asarray(result["unfolded"], dtype=float) for result in self.normalized_results]
        )
        varied_flat = {}
        for systematic in self.systematics:
            if systematic == "nominal":
                continue
            varied_flat[systematic] = np.concatenate(
                [
                    np.asarray(per_pt["unfolded"][systematic], dtype=float)
                    for per_pt in self.normalized_systematics
                ]
            )

        cov_syst = np.zeros((len(nominal_flat), len(nominal_flat)))
        seen = set()
        use_model_envelope = getattr(self.spec, "model_envelope", False)
        model_superseded = ("herwig", "fsr", "isr", "model_")
        for systematic in varied_flat:
            if use_model_envelope and systematic.startswith(model_superseded):
                continue
            if systematic in seen:
                continue
            if systematic.endswith("Up") or systematic.endswith("Down"):
                source = systematic[:-2] if systematic.endswith("Up") else systematic[:-4]
                up = varied_flat.get(source + "Up")
                down = varied_flat.get(source + "Down")
                seen.update({source + "Up", source + "Down"} & varied_flat.keys())
                if up is not None and down is not None:
                    shift = 0.5 * (up - down)
                else:
                    shift = (up if up is not None else down) - nominal_flat
            else:
                seen.add(systematic)
                shift = varied_flat[systematic] - nominal_flat
            cov_syst += np.outer(shift, shift)
        if use_model_envelope:
            cov_syst += self._normalized_model_covariance(nominal_flat)
        return cov_syst


    def _normalized_model_covariance(self, nominal_flat):
        """Coherent two-leg model covariance in normalized-result space."""

        nominal_flat = np.asarray(nominal_flat, dtype=float)
        if self._uses_enclosing_model_covariance():
            return sum(self.model_group_covariances.values())
        ps_frac = getattr(self, "model_ps_shift_flat", None)
        had_frac = getattr(self, "model_had_shift_flat", None)
        if ps_frac is None or had_frac is None:
            return np.zeros((nominal_flat.size, nominal_flat.size), dtype=float)

        scope = getattr(self.spec, "model_covariance_scope", "global_shown")
        if scope == "per_pt":
            fractional_vectors = [
                vector
                for pair in zip(
                    self.model_ps_shifts_by_pt_flat,
                    self.model_had_shifts_by_pt_flat,
                )
                for vector in pair
            ]
        elif scope in ("global_shown", "global_all"):
            fractional_vectors = [ps_frac, had_frac]
        else:
            raise ValueError(f"Unknown model_covariance_scope: {scope}")

        offsets = np.concatenate(
            [[0], np.cumsum([len(edges) - 1 for edges in self.gen_edges_by_pt])]
        ).astype(int)
        covariance = np.zeros((nominal_flat.size, nominal_flat.size), dtype=float)
        for vector_frac in fractional_vectors:
            vector = np.asarray(vector_frac, dtype=float) * nominal_flat
            for i in range(len(self.gen_edges_by_pt)):
                block = slice(offsets[i], offsets[i + 1])
                mask = self._shown_gen_mask(i)
                widths = np.diff(np.asarray(self.gen_edges_by_pt[i], dtype=float))
                y_slice = nominal_flat[block]
                normalization = float(np.sum(widths[mask] * y_slice[mask]))
                if normalization == 0.0:
                    continue
                leak = float(np.sum(widths[mask] * vector[block][mask]))
                vector[block] = np.where(
                    mask,
                    vector[block] - (leak / normalization) * y_slice,
                    vector[block],
                )
            covariance += np.outer(vector, vector)
        return covariance


    def _uses_enclosing_model_covariance(self):
        spec = getattr(self, "spec", None)
        return (getattr(spec, "model_envelope", False)
                and getattr(spec, "model_covariance_method", "selected_variation") == "enclosing_ellipsoid")


    def _compute_enclosing_model_covariances(self):
        """Fit both model groups using the complete normalized templates."""
        from unfold.model import MODEL_GROUPS, two_group_model_covariance

        if self.spec.model_envelope_source != "prepared_systematics":
            raise ValueError("Enclosing model covariance currently requires prepared pair-split variations")
        if self.spec.model_covariance_scope != "global_templates":
            raise ValueError("Enclosing model covariance requires scope='global_templates'")
        nominal = np.concatenate([result["unfolded"] for result in self.normalized_results])
        varied = {
            source: np.concatenate([result["unfolded"][source] for result in self.normalized_systematics])
            for sources in MODEL_GROUPS.values() for source in sources
        }
        weights = np.zeros((len(self.gen_edges_by_pt), nominal.size))
        offset = 0
        for i, edges in enumerate(self.gen_edges_by_pt):
            block = slice(offset, offset + len(edges) - 1)
            weights[i, block] = np.diff(edges) * self._shown_gen_mask(i)
            offset = block.stop
        self.model_group_covariances, self.model_covariance_diagnostics = two_group_model_covariance(
            nominal, varied, weights)
        self._results_chi2_cov = None


    def get_total_covariance(self):
        """Total covariance of the normalized result (stat + systematics)."""
        return self.norm_cov_stat + self.get_systematic_covariance()

    # Pairs "<base>Up[_corr|_uncorr_YYYY]" with the matching Down key. The
    # Up/Down must sit immediately before the era suffix: a naive
    # replace("Up", "Down") would corrupt e.g. JES_PileUpPtBB.
    _UPDOWN_KEY = re.compile(r"^(.+)(Up|Down)((?:_corr|_uncorr_\d+)?)$")


    def _results_chi2_covariance(self):
        """Covariance of the normalized result for the quoted data-vs-MC chi2.

        Flat over all gen bins (normalized, per-bin-width units), cached.
        Three terms:

        * statistics: ``norm_cov_stat`` (TUnfold input + matrix-stat
          covariances through the normalization Jacobian);
        * detector systematics: rank-1 outer products of the normalized
          shifts, symmetrized (up-down)/2 when both legs exist. Sources
          superseded by the model envelope (herwig/fsr/isr, and the raw
          ``model_*`` variations) are excluded when the envelope is active,
          mirroring the plotted band;
        * modelling: the coherent PS/HAD shift vectors of the bottom-line
          construction, scaled to the normalized nominal and **projected onto
          the normalization-preserving subspace** of each pT slice
          (w . s = 0 over the shown bins, w = bin widths). The stored
          fractional shifts carry a rate-like component that a normalized
          measurement cannot fluctuate in; every detector source loses that
          component by construction (each variation is renormalized by its
          own sum), so the model term must as well.

        Within each slice the sum constraint leaves one exact null direction,
        so consumers must use a pseudo-inverse and ndof = n_bins - 1.
        """
        cached = getattr(self, "_results_chi2_cov", None)
        if cached is not None:
            return cached

        nominal_flat = np.concatenate(
            [np.asarray(r["unfolded"], dtype=float) for r in self.normalized_results]
        )
        covariance = np.array(self.norm_cov_stat, copy=True)

        use_model_envelope = getattr(self.spec, "model_envelope", False)
        superseded = ("herwig", "fsr", "isr", "model_") if use_model_envelope else ("model_",)
        shifts = {
            syst: np.concatenate(
                [
                    np.asarray(per_pt["unfolded"][syst], dtype=float)
                    for per_pt in self.normalized_systematics
                ]
            )
            for syst in self.systematics
            if syst != "nominal" and not syst.startswith(superseded)
        }
        done = set()
        for syst, varied in shifts.items():
            if syst in done:
                continue
            match = self._UPDOWN_KEY.match(syst)
            partner = None
            if match:
                base, direction, suffix = match.groups()
                other = "Down" if direction == "Up" else "Up"
                partner = f"{base}{other}{suffix}"
            if partner is not None and partner in shifts:
                done.update({syst, partner})
                up = varied if match.group(2) == "Up" else shifts[partner]
                down = shifts[partner] if match.group(2) == "Up" else varied
                shift = 0.5 * (up - down)
            else:
                done.add(syst)
                shift = varied - nominal_flat
            covariance += np.outer(shift, shift)

        covariance += self._normalized_model_covariance(nominal_flat)

        self._results_chi2_cov = covariance
        return covariance


    def save_normalized_covariance(self):
        """Write the normalized-result covariances to an NPZ next to the plots."""
        suffix = "groomed" if self.groomed else "ungroomed"
        nominal_flat = np.concatenate(
            [np.asarray(result["unfolded"], dtype=float) for result in self.normalized_results]
        )
        cov_syst = self.get_systematic_covariance()
        save_path = self._relocate_output(f"normalized_covariance_{suffix}.npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            normalized=nominal_flat,
            cov_stat_input=self.norm_cov_input,
            cov_stat_matrix=self.norm_cov_matrix,
            cov_stat=self.norm_cov_stat,
            cov_syst=cov_syst,
            cov_total=self.norm_cov_stat + cov_syst,
            pt_edges=np.asarray(self.pt_edges, dtype=float),
            gen_bins_per_pt=np.asarray(
                [len(edges) - 1 for edges in self.gen_edges_by_pt], dtype=int
            ),
            tau=float(self.tau or 0.0),
            **({"cov_model_ps": self.model_group_covariances["parton_shower"],
                "cov_model_had": self.model_group_covariances["hadronization"]}
               if self._uses_enclosing_model_covariance() else {}),
        )
        print(f"Saved normalized covariances to {save_path}")


    def _compute_total_systematic(self):
        print("Computing total systematic uncertainty...")
        self._compute_normalized_stat_covariance()
        use_jacobian = self.stat_propagation == "jacobian"
        if use_jacobian:
            input_std_pt_binned = unflatten_gen_by_pt(
                np.sqrt(np.clip(np.diag(self.norm_cov_input), 0.0, None)),
                self.gen_edges_by_pt,
            )
            matrix_std_pt_binned = unflatten_gen_by_pt(
                np.sqrt(np.clip(np.diag(self.norm_cov_matrix), 0.0, None)),
                self.gen_edges_by_pt,
            )
        # Compute total systematic uncertainty for each pt bin
        # load from file
        #herwig_unc = np.load("./inputs/zjet/mass/herwig_closure_unc_mass_groomed.npy") if self.groomed else np.load("./inputs/zjet/mass/herwig_closure_unc_mass_ungroomed.npy")
        #self.herwig_unc = herwig_unc
        use_model_envelope = getattr(self.spec, "model_envelope", False)
        if use_model_envelope:
            # Vincia/CR/frag model-response variations.  Z+jet builds them
            # offline from its fine 2018 response; prepared pair-split inputs
            # provide equivalent model_<source> response categories built
            # before the candidate GEN-bin merge.  FSR joins either path below
            # from the stored PSWeight unfolds.  This supersedes the legacy
            # herwig-difference term and the raw fsr/isr quadrature entries.
            from unfold.model import (
                compute_model_shifts,
                compute_prepared_model_shifts,
                group_model_shifts,
            )
            envelope_source = getattr(
                self.spec, "model_envelope_source", "zjet_offline"
            )
            if envelope_source == "zjet_offline":
                print("Computing model envelope (Vincia/CR/frag column-scaled responses)...")
                raw_shifts = compute_model_shifts(self)
            elif envelope_source == "prepared_systematics":
                if not getattr(self, "_uses_prepared_inputs", False):
                    raise ValueError(
                        "model_envelope_source='prepared_systematics' requires prepared inputs"
                    )
                print("Computing model envelope from prepared model response variations...")
                raw_shifts = compute_prepared_model_shifts(self)
            else:
                raise ValueError(
                    f"unsupported model_envelope_source {envelope_source!r}"
                )
            self.model_shift_components = group_model_shifts(
                raw_shifts, len(self.gen_edges_by_pt))
            self.model_signed_shifts = raw_shifts
            self.model_signed_shifts["fsrUp"] = {}
            self.model_signed_shifts["fsrDown"] = {}
            if self._uses_enclosing_model_covariance():
                self._compute_enclosing_model_covariances()
        # Systematic-axis entries folded into the model envelope instead of the
        # quadrature sum when model_envelope is on.
        _model_superseded = ("herwig", "fsr", "isr", "model_")
        for i in range(len(self.normalized_results)):
            nominal = self.normalized_results[i]['unfolded']
            syst_up_total = np.zeros_like(nominal)
            syst_down_total = np.zeros_like(nominal)

            for syst in self.systematics:
                if syst == 'nominal':
                    continue
                if use_model_envelope and syst.startswith(_model_superseded):
                    continue
                varied = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                diff_sq = np.abs(varied - nominal) ** 2
                if 'Down' in syst:
                    syst_down_total += diff_sq
                else:
                    syst_up_total += diff_sq
            if use_model_envelope:
                # Model term: per-bin max over {Vincia, CR, frag} re-unfold
                # shifts and the FSR PSWeight shift, added symmetrically.
                # Bins holding <0.5% of the normalized slice are below
                # measurement sensitivity (e.g. the ungroomed [-10,-2.5]
                # catch-all, ~0.3-1% and consistent with zero): a fractional
                # shift there is noise-over-noise, so the model fraction is
                # zeroed (its absolute band contribution is negligible either
                # way).
                widths = np.diff(np.asarray(self.gen_edges_by_pt[i], float))
                content = np.abs(nominal) * widths
                # positive density AND >0.5% of the slice: negative unfolded
                # bins are consistent with zero, so no relative uncertainty
                # is defined there.
                sensitive = (np.asarray(nominal) > 0) & (
                    content > 5e-3 * max(content.sum(), 1e-300))
                if self._uses_enclosing_model_covariance():
                    # Keep complete normalized shifts; binwise censoring would
                    # break the normalization constraint and remove directions.
                    sensitive = np.ones_like(sensitive, dtype=bool)
                fsr_frac = np.zeros_like(nominal)
                for var in ("fsrUp", "fsrDown"):
                    varied = self.normalized_systematics[i]['unfolded'].get(
                        var, np.zeros_like(nominal))
                    if np.any(varied):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            signed_frac = np.divide(
                                varied - nominal, nominal,
                                out=np.zeros_like(nominal), where=nominal != 0)
                        self.model_signed_shifts[var][i] = np.where(
                            sensitive, signed_frac, 0.0)
                        fsr_frac = np.maximum(fsr_frac, np.abs(signed_frac))
                    else:
                        self.model_signed_shifts[var][i] = np.zeros_like(nominal)
                fsr_frac = np.where(sensitive, fsr_frac, 0.0)
                for name in ("Vincia", "CR", "frag"):
                    comp = np.where(
                        sensitive, self.model_shift_components[name][i], 0.0)
                    self.model_shift_components[name][i] = comp
                for source in ("vincia", "cr1", "cr2", "fraghard", "fragsoft"):
                    self.model_signed_shifts[source][i] = np.where(
                        sensitive, self.model_signed_shifts[source][i], 0.0)
                # ARC round-2 prescription (plan WS2): the shower and the
                # hadronization modelling are two INDEPENDENT uncertainties,
                # each an envelope over its own variations, combined in
                # quadrature -- never one max envelope over everything.
                #   PS leg  = Vincia shower swap (hadronization kept nominal),
                #             enveloped with the FSR PSWeight scale variation;
                #   HAD leg = hadronization-parameter variations on the default
                #             shower (CR modes 1/2, Lund frag hard/soft).
                ps_frac = np.maximum(
                    fsr_frac, self.model_shift_components["Vincia"][i])
                had_frac = np.maximum(
                    self.model_shift_components["CR"][i],
                    self.model_shift_components["frag"][i])
                model_frac = np.sqrt(ps_frac**2 + had_frac**2)
                self.normalized_results[i]['model_envelope_ps_frac'] = ps_frac.copy()
                self.normalized_results[i]['model_envelope_had_frac'] = had_frac.copy()
                if self._uses_enclosing_model_covariance():
                    offset = sum(len(edges) - 1 for edges in self.gen_edges_by_pt[:i])
                    block = slice(offset, offset + len(nominal))
                    ps_unc = np.sqrt(np.clip(np.diag(self.model_group_covariances["parton_shower"])[block], 0, None))
                    had_unc = np.sqrt(np.clip(np.diag(self.model_group_covariances["hadronization"])[block], 0, None))
                    ps_frac = np.divide(ps_unc, np.abs(nominal), out=np.full_like(ps_unc, np.nan), where=nominal != 0)
                    had_frac = np.divide(had_unc, np.abs(nominal), out=np.full_like(had_unc, np.nan), where=nominal != 0)
                    model_frac = np.hypot(ps_frac, had_frac)
                if not hasattr(self, "model_fsr_frac"):
                    self.model_fsr_frac = {}
                if not hasattr(self, "model_ps_frac"):
                    self.model_ps_frac = {}
                    self.model_had_frac = {}
                self.model_fsr_frac[i] = fsr_frac
                self.model_ps_frac[i] = ps_frac
                self.model_had_frac[i] = had_frac
                model_unc = model_frac * np.abs(nominal)
                if self._uses_enclosing_model_covariance():
                    model_unc = np.hypot(ps_unc, had_unc)
                syst_up_total += model_unc**2
                syst_down_total += model_unc**2
                self.normalized_results[i]['model_unc_frac'] = model_frac
                self.normalized_results[i]['model_ps_frac'] = ps_frac
                self.normalized_results[i]['model_had_frac'] = had_frac
            if use_jacobian:
                # Errors of the normalized result itself: the Jacobian removes
                # the fluctuation common to all bins of a pT slice.
                input_stat_unc = input_std_pt_binned[i]
                matrix_stat_unc = matrix_std_pt_binned[i]
                with np.errstate(divide='ignore', invalid='ignore'):
                    input_stat_frac = np.abs(np.divide(
                        input_stat_unc, nominal,
                        out=np.zeros_like(input_stat_unc), where=nominal != 0,
                    ))
                    matrix_stat_frac = np.abs(np.divide(
                        matrix_stat_unc, nominal,
                        out=np.zeros_like(matrix_stat_unc), where=nominal != 0,
                    ))
                stat_frac = np.sqrt(input_stat_frac**2 + matrix_stat_frac**2)
            else:
                input_stat_unc = self.input_stat_unc_pt_binned[i] * nominal
                matrix_stat_unc = self.matrix_stat_unc_pt_binned[i] * nominal
                input_stat_frac = self.input_stat_unc_pt_binned[i]
                matrix_stat_frac = self.matrix_stat_unc_pt_binned[i]
                stat_frac = self.stat_unc_pt_binned[i]
            stat_unc = np.sqrt(input_stat_unc**2 + matrix_stat_unc**2)

            syst_up_total += stat_unc**2
            syst_down_total += stat_unc**2
            # Take sqrt of sum of squares for total uncertainty
            total_up_unc = np.sqrt(syst_up_total)
            total_down_unc = np.sqrt(syst_down_total)
            self.normalized_results[i]['syst_unc'] = {
            'up': total_up_unc,
            'down': total_down_unc
            }
            self.normalized_results[i]['input_stat_unc_frac'] = input_stat_frac
            self.normalized_results[i]['matrix_stat_unc_frac'] = matrix_stat_frac
            self.normalized_results[i]['stat_unc_frac'] = stat_frac
            self.normalized_results[i]['input_stat_unc'] = input_stat_unc
            self.normalized_results[i]['matrix_stat_unc'] = matrix_stat_unc
            self.normalized_results[i]['stat_unc'] = stat_unc

        # Flat per-gen-bin model-uncertainty fraction, concatenated over pT
        # slices in the same order as gen_edges_by_pt -> used by the bottom-line
        # test to fold the modelling uncertainty into the unfolded-data
        # covariance (ARC round-2 request). Zeros when no model envelope.
        if use_model_envelope and self._uses_enclosing_model_covariance():
            # A group covariance has several possible directions; do not
            # advertise one selected source or serialize a fictitious vector.
            self.model_unc_frac_flat = np.concatenate([
                result['model_unc_frac'] for result in self.normalized_results])
            self.model_ps_source = "enclosing templates"
            self.model_had_source = "enclosing templates"
            self.model_ps_sources_by_pt = {}
            self.model_had_sources_by_pt = {}
        elif use_model_envelope:
            self.model_unc_frac_flat = np.concatenate([
                np.asarray(
                    self.normalized_results[i].get(
                        "model_unc_frac",
                        np.zeros(len(self.gen_edges_by_pt[i]) - 1)),
                    float)
                for i in range(len(self.normalized_results))
            ])
            # Preserve physical bin-to-bin correlations in the bottom-line
            # covariance. Select one coherent, signed variation for each model
            # leg using its global L2 norm over the full unfolded space; never
            # splice a different envelope source into each bin.
            def _flat_signed(source):
                return np.concatenate([
                    np.asarray(self.model_signed_shifts[source][i], float)
                    for i in range(len(self.normalized_results))
                ])

            ps_candidates = {
                source: _flat_signed(source)
                for source in ("vincia", "fsrUp", "fsrDown")
            }
            had_candidates = {
                source: _flat_signed(source)
                for source in ("cr1", "cr2", "fraghard", "fragsoft")
            }
            # Source selection mask: the production prescription uses exactly
            # the publication phase space. The 185-200 migration bin and the
            # hidden low-rho buffers remain in the response but cannot decide
            # which variation represents the reported covariance.
            shown_mask_parts = []
            shown_local_masks = []
            floors = self._bl_shown_floors()
            for i, edges in enumerate(self.gen_edges_by_pt):
                lower_edges = np.asarray(edges[:-1], float)
                local = ((self.pt_edges[i] >= 200)
                         & (lower_edges >= floors[i] - 1e-9))
                shown_local_masks.append(local)
                shown_mask_parts.append(local)
            shown_mask = np.concatenate(shown_mask_parts)
            scope = getattr(
                self.spec, "model_covariance_scope", "global_shown")
            selection_mask = (
                np.ones_like(shown_mask, dtype=bool)
                if scope == "global_all" else shown_mask
            )

            def _selected_norm(vector, mask):
                return np.linalg.norm(np.asarray(vector)[mask])

            self.model_ps_source = max(
                ps_candidates,
                key=lambda source: _selected_norm(
                    ps_candidates[source], selection_mask))
            self.model_had_source = max(
                had_candidates,
                key=lambda source: _selected_norm(
                    had_candidates[source], selection_mask))
            self.model_ps_shift_flat = ps_candidates[self.model_ps_source]
            self.model_had_shift_flat = had_candidates[self.model_had_source]

            # Diagnostic prescription with independent PS/HAD nuisances per
            # published pT slice. Each vector is embedded in the full flat
            # space, making _model_cov_unfolded block diagonal across pT.
            offsets = np.concatenate([
                [0], np.cumsum([
                    len(edges) - 1 for edges in self.gen_edges_by_pt
                ])
            ]).astype(int)
            self.model_ps_shifts_by_pt_flat = []
            self.model_had_shifts_by_pt_flat = []
            self.model_ps_sources_by_pt = {}
            self.model_had_sources_by_pt = {}
            for i, local_mask in enumerate(shown_local_masks):
                block = slice(offsets[i], offsets[i + 1])
                if self.pt_edges[i] < 200 or not np.any(local_mask):
                    continue
                ps_source = max(
                    ps_candidates,
                    key=lambda source: np.linalg.norm(
                        ps_candidates[source][block][local_mask]))
                had_source = max(
                    had_candidates,
                    key=lambda source: np.linalg.norm(
                        had_candidates[source][block][local_mask]))
                ps_vector = np.zeros_like(self.model_ps_shift_flat)
                had_vector = np.zeros_like(self.model_had_shift_flat)
                ps_vector[block] = ps_candidates[ps_source][block]
                had_vector[block] = had_candidates[had_source][block]
                self.model_ps_shifts_by_pt_flat.append(ps_vector)
                self.model_had_shifts_by_pt_flat.append(had_vector)
                self.model_ps_sources_by_pt[i] = ps_source
                self.model_had_sources_by_pt[i] = had_source
            print(
                "Bottom-line model covariance: correlated coherent shifts "
                f"scope={scope}, PS={self.model_ps_source}, "
                f"HAD={self.model_had_source}"
            )
        else:
            self.model_unc_frac_flat = None
            self.model_ps_shift_flat = None
            self.model_had_shift_flat = None


    # systematic naming (see unfold.systematics)
    def _get_systematic_group_name(self, syst_name):
        return systnames.group_name(syst_name)

    def _get_systematic_label(self, syst_name):
        return systnames.label(syst_name)

    def _split_systematic_variation(self, syst_name):
        return systnames.split_updown(syst_name)

    def _get_systematic_summary_name(self, syst_name, grouped=False):
        return systnames.summary_name(syst_name, grouped=grouped)

    def _build_syst_fraction_dict(self, pt_index):
        result = self.normalized_results[pt_index]
        nominal = result["unfolded"]
        total_syst_up = result["syst_unc"]["up"]
        total_syst_down = result["syst_unc"]["down"]
        syst_fraction_dict = {}
        use_model_envelope = getattr(self.spec, "model_envelope", False)

        for syst_name, syst_unfolded in self.normalized_systematics[pt_index]["unfolded"].items():
            # With the model envelope on, herwig/fsr/isr are superseded in the
            # total (see _compute_total_systematic); showing their legacy
            # curves would misrepresent the decomposition.
            if use_model_envelope and syst_name.startswith(
                ("herwig", "fsr", "isr", "model_")
            ):
                continue
            diff = syst_unfolded - nominal
            syst_fraction = np.abs(np.divide(diff, nominal, out=np.zeros_like(diff), where=nominal != 0))
            syst_fraction_dict[syst_name] = syst_fraction

        if use_model_envelope and "model_ps_frac" in result:
            # The two independent modelling legs entering the total in
            # quadrature (ARC round-2): shower (Vincia + FSR envelope) and
            # hadronization (CR/frag envelope). Shown separately.
            syst_fraction_dict["showermodelUp"] = np.asarray(
                result["model_ps_frac"], float)
            syst_fraction_dict["hadmodelUp"] = np.asarray(
                result["model_had_frac"], float)
        elif use_model_envelope and "model_unc_frac" in result:
            syst_fraction_dict["modelenvelopeUp"] = np.asarray(
                result["model_unc_frac"], float)

        # The summary is drawn for the *normalized* unfolded result, so its
        # statistical component must come from the same normalization-Jacobian
        # propagation as ``syst_unc``.  ``self.stat_unc_pt_binned`` is the
        # pre-normalization relative error and can be larger than the normalized
        # total in bins where the per-pT area constraint removes a common mode.
        stat_fraction = np.asarray(result["stat_unc_frac"], dtype=float)
        total_syst_fraction_up = np.abs(np.divide(total_syst_up, np.abs(nominal), out=np.zeros_like(total_syst_up), where=np.abs(nominal) != 0))
        total_syst_fraction_down = np.abs(np.divide(total_syst_down, np.abs(nominal), out=np.zeros_like(total_syst_down), where=np.abs(nominal) != 0))

        tolerance = 1e-12 * np.maximum(
            1.0,
            np.maximum(total_syst_fraction_up, total_syst_fraction_down),
        )
        if np.any(stat_fraction > total_syst_fraction_up + tolerance) or np.any(
            stat_fraction > total_syst_fraction_down + tolerance
        ):
            raise RuntimeError(
                "normalized total uncertainty is smaller than its statistical component"
            )

        syst_fraction_dict["Stat Unc"] = stat_fraction
        syst_fraction_dict["Total_Up"] = total_syst_fraction_up
        syst_fraction_dict["Total_Down"] = total_syst_fraction_down
        return syst_fraction_dict


    def _group_syst_fraction_dict(self, syst_fraction_dict, grouped=True):
        grouped_fraction_dict = {}
        accumulators = {}

        for syst_name, syst_fraction in syst_fraction_dict.items():
            if syst_name in {"Stat Unc", "Total_Up", "Total_Down"}:
                continue

            target_name = self._get_systematic_summary_name(syst_name, grouped=grouped)
            _, variation = self._split_systematic_variation(syst_name)
            target_key = f"{target_name}Down" if variation == "Down" else f"{target_name}Up"

            if target_key not in accumulators:
                accumulators[target_key] = np.zeros_like(syst_fraction)
            accumulators[target_key] += syst_fraction**2

        for target_key, target_sum in accumulators.items():
            grouped_fraction_dict[target_key] = np.sqrt(target_sum)

        grouped_fraction_dict["Stat Unc"] = syst_fraction_dict["Stat Unc"]
        grouped_fraction_dict["Total_Up"] = syst_fraction_dict["Total_Up"]
        grouped_fraction_dict["Total_Down"] = syst_fraction_dict["Total_Down"]
        return grouped_fraction_dict


    def _resolve_raw_systematic_pair(self, syst_name):
        if syst_name in self.systematics:
            base_name, variation = self._split_systematic_variation(syst_name)
            if variation == "Up":
                return syst_name, f"{base_name}Down"
            if variation == "Down":
                return f"{base_name}Up", syst_name

        candidates = [name for name in self.systematics if name != "nominal"]
        matched_up = None
        matched_down = None

        for candidate in candidates:
            base_name, variation = self._split_systematic_variation(candidate)
            if base_name.lower() != syst_name.lower():
                continue
            if variation == "Up":
                matched_up = candidate
            elif variation == "Down":
                matched_down = candidate

        return matched_up, matched_down


    def _correlation_covariance(self, covariance):
        """Return the requested covariance basis for a correlation plot."""
        if covariance not in {"stat", "total"}:
            raise ValueError("covariance must be either 'stat' or 'total'")
        if covariance == "total":
            return np.array(self.get_total_covariance(), copy=True)
        if self.stat_propagation == "jacobian":
            # Correlation of the normalized result: stat covariance propagated
            # through the normalization Jacobian (negative correlations from
            # the per-pT-slice sum constraint are expected).
            return np.array(self.norm_cov_stat, copy=True)
        return self.cov_uncorr_np + self.cov_data_np


    def _crop_matrix_view_to_floor(self, matrix, reco_by, gen_by, floor):
        """Drop per-pT rho bins below ``floor`` from both axes of a reported
        response-matrix view (display only). Returns the cropped matrix and the
        trimmed per-pT edge lists; the unfolding itself is unaffected."""
        def mask_and_edges(edges_by_pt):
            masks, new_edges = [], []
            for e in edges_by_pt:
                e = np.asarray(e, float)
                keep = e[:-1] >= floor - 1e-9      # bins whose lower edge >= floor
                masks.append(keep)
                idx = np.flatnonzero(keep)
                new_edges.append(list(e[idx[0]:idx[-1] + 2]) if idx.size else list(e[-1:]))
            return np.concatenate(masks), new_edges

        rmask, new_reco = mask_and_edges(reco_by)
        gmask, new_gen = mask_and_edges(gen_by)
        cropped = np.asarray(matrix)[np.ix_(rmask, gmask)]
        return cropped, new_reco, new_gen


    def _gen_binned_migration(self, matrix=None):
        """Compress the matched response to gen binning on both axes.

        Returns A with A[j, k] = matched events reconstructed in the gen-bin
        region j and generated in gen bin k (reco mass bins are grouped into
        the gen bins of the same pT slice; pT edges are shared). ``matrix``
        defaults to the nominal mosaic; pass its variance mosaic to compress
        sumw2 the same way (grouping = summing, so variances stay valid).
        """
        mosaic = self.mosaic_dict["nominal"] if matrix is None else matrix
        n_gen = mosaic.shape[1]
        compressed = np.zeros((n_gen, n_gen))
        reco_offset = 0
        gen_offset = 0
        for i, gen_edges in enumerate(self.gen_edges_by_pt):
            reco_edges = np.asarray(self.reco_edges_by_pt[i], dtype=float)
            centers = 0.5 * (reco_edges[:-1] + reco_edges[1:])
            gen_edges = np.asarray(gen_edges, dtype=float)
            for k in range(len(gen_edges) - 1):
                rows = np.flatnonzero(
                    (centers >= gen_edges[k]) & (centers < gen_edges[k + 1])
                ) + reco_offset
                compressed[gen_offset + k, :] += mosaic[rows, :].sum(axis=0)
            reco_offset += len(reco_edges) - 1
            gen_offset += len(gen_edges) - 1
        return compressed


    def _compute_input_stat_unc_from_covariance(self):
        """Read absolute data/response errors from the selected covariances."""

        input_variance = np.clip(np.diag(self.cov_data_np), 0.0, None)
        input_std = np.sqrt(input_variance)
        cov_uncorr = getattr(self, "cov_uncorr_np", None)
        if cov_uncorr is None:
            cov_uncorr = np.zeros_like(self.cov_data_np)
        matrix_variance = np.clip(np.diag(cov_uncorr), 0.0, None)
        matrix_std = np.sqrt(matrix_variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.input_stat_unc_frac = np.abs(
                np.divide(
                    input_std,
                    self.y_unf,
                    out=np.zeros_like(input_std),
                    where=self.y_unf != 0,
                )
            )
            self.matrix_stat_unc_frac = np.abs(
                np.divide(
                    matrix_std,
                    self.y_unf,
                    out=np.zeros_like(matrix_std),
                    where=self.y_unf != 0,
                )
            )
        self.stat_unc_frac = np.sqrt(
            self.input_stat_unc_frac**2 + self.matrix_stat_unc_frac**2
        )
        self.input_stat_unc_pt_binned = unflatten_gen_by_pt(
            self.input_stat_unc_frac,
            self.gen_edges_by_pt,
        )
        self.matrix_stat_unc_pt_binned = unflatten_gen_by_pt(
            self.matrix_stat_unc_frac,
            self.gen_edges_by_pt,
        )
        self.stat_unc_pt_binned = unflatten_gen_by_pt(
            self.stat_unc_frac,
            self.gen_edges_by_pt,
        )
