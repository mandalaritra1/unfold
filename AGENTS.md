# Unfold repository instructions

This repository implements CMS jet-mass unfolding.  Preserve unrelated work and
keep requested changes small.  Explain any effect on physics results.

Layout and commands: `README.md` (top level) and `scripts/README.md`.  The
package is `src/unfold/`; `legacy/` is the pre-restructure tree and is
reference only, never imported.

Preserve selections, weights, corrections, binning, response definitions,
normalization, and uncertainty conventions unless the requested change
authorizes otherwise.  Resolve material physics ambiguity before changing
behavior.  The Z+jet JES year-correlation split changed to the JetMET sqrt prescription
on 2026-09-09; the Z+jet golden regression runs with `--era-split linear`
because its reference tree predates that (README "Physics changes and
caveats").  The legacy herwig fallback projection is frozen on purpose.

Every channel runs through `unfold run --channel C [--observable O] [--tag T]`;
`original` is the production tag.  Before an unfolding run verify the inputs
exist.  After a change to the
engine, the loaders or the binnings, rerun the affected golden case and
compare with `tests/compare_golden.py` (README "Checking that nothing
changed").  For plot changes, inspect the affected renders.

For analysis interpretation or workflow history consult
`~/Projects/ai-wiki/wiki/repos/unfold.md`; read that wiki's `AGENTS.md` before
editing it.  Record confirmed reusable findings there.  Treat raw sources and
other linked repositories as read-only unless authorized.
