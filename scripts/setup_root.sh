#!/usr/bin/env bash
# Source the ROOT build that is compatible with this repository's Python venv.
#
# Usage:
#   source scripts/setup_root.sh

UNFOLD_ROOTSYS="${UNFOLD_ROOTSYS:-/Users/aritra/opt/root-6.40.00-rc1}"

if [ ! -f "${UNFOLD_ROOTSYS}/bin/thisroot.sh" ]; then
  echo "ROOT setup not found: ${UNFOLD_ROOTSYS}/bin/thisroot.sh" >&2
  return 1 2>/dev/null || exit 1
fi

_unfold_root_old_pwd="$(pwd)"
cd "${UNFOLD_ROOTSYS}" || return 1 2>/dev/null || exit 1
source bin/thisroot.sh
cd "${_unfold_root_old_pwd}" || return 1 2>/dev/null || exit 1
unset _unfold_root_old_pwd
