# Model-variation gen reweighters (24-bin, w = variation / nominal CP5)

Copied from `smp_jetmass_run2/smp_jetmass_run2/corrections/` (commit dd0bd4d era).
- vincia: derived from the standalone Vincia vs CP5 production (same-ME path B).
- cr1/cr2, fraghard/fragsoft: from the pythia_var HT-binned NanoGEN
  (`/eos/user/a/amandal/pythia_var`, ~15.5k selected events total).

Format: PtVarWeighter npz — pt_edges [0,200,290,400,13000], per-pt `rho_grids`
(doubled-edge step grid) + `w_grids`. w is piecewise-constant on the 24-bin gen
rho axis (fine 48-bin axis merged in pairs), so scaling the gen columns of the
nominal response by w is EXACTLY equivalent to the per-event reskim (validated
2026-07-07 at machine precision, and against a full UL18 casa reskim).
Used by `unfold.tools.model_envelope` for the production model uncertainty.
