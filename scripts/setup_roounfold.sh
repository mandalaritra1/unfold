#!/usr/bin/env bash
# Point the unfold package at a built RooUnfold shared library so the
# D'Agostini (RooUnfoldBayes) backend can be loaded in PyROOT.
#
# Usage:
#   source scripts/setup_root.sh        # ROOT must be set up first
#   source scripts/setup_roounfold.sh
#
# RooUnfold is an external dependency (not pip). Build it once against the same
# ROOT as scripts/setup_root.sh, e.g.:
#   git clone https://gitlab.cern.ch/RooUnfold/RooUnfold.git ~/opt/RooUnfold
#   cd ~/opt/RooUnfold
#   make -f GNUmakefile shlib \
#     ROOTLIBS="$(root-config --libs) -lUnfold -lXMLParser -lXMLIO -lRooFitJSONInterface"
# (the extra libs resolve RooUnfold's bundled TUnfold-XML and RooFit-JSON deps).

UNFOLD_ROOUNFOLD_LIB="${UNFOLD_ROOUNFOLD_LIB:-$HOME/opt/RooUnfold/libRooUnfold}"

_candidate="${UNFOLD_ROOUNFOLD_LIB}"
if [ ! -f "${_candidate}" ] && [ ! -f "${_candidate}.so" ] && [ ! -f "${_candidate}.dylib" ]; then
  echo "RooUnfold library not found at: ${_candidate}{,.so,.dylib}" >&2
  echo "Build it (see the header of this script) or set UNFOLD_ROOUNFOLD_LIB." >&2
  return 1 2>/dev/null || exit 1
fi

export UNFOLD_ROOUNFOLD_LIB
echo "UNFOLD_ROOUNFOLD_LIB=${UNFOLD_ROOUNFOLD_LIB}"
