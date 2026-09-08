# Unfold repository instructions

This repository implements CMS jet-mass unfolding. Preserve unrelated work and
keep requested changes small. Explain any effect on physics results.

For analysis interpretation or workflow changes, consult the relevant sections of
`~/Projects/ai-wiki/wiki/repos/unfold.md`. Search `wiki/meta/index.md` when
additional context is needed. Small edits do not require the full wiki or repo card.
Read the wiki's `AGENTS.md` before editing it.

Preserve selections, weights, corrections, binning, response definitions,
normalization, and uncertainty conventions unless the requested change authorizes
otherwise. Resolve material physics ambiguity before changing behavior; routine
implementation and local validation within an established choice may continue.

Use `README.md` and `scripts/README.md` for environment and workflow commands.
Run checks appropriate to the changed behavior. Repeat or broaden checks when a
change, failure, or unresolved concern justifies it. Verify required inputs before
an unfolding run. For plot changes, inspect affected renders.

Record confirmed reusable findings in the existing ai-wiki pages when warranted.
Follow its index/log conventions for wiki edits. Routine passing checks, unchanged
status, and cosmetic edits do not require a new knowledge entry. Treat raw sources
and other linked repositories as read-only unless authorized.
