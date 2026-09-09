#!/usr/bin/env bash
# One-shot environment for this repository.  Usage, from the repo root:
#
#     source env.sh
#
# It creates .venv on first use (Python 3.11, the version the local ROOT build
# was compiled against), installs the package into it in editable mode,
# activates the venv, and puts ROOT (and RooUnfold, if built) on the path.
# After that `unfold`, `python` and `pytest` all refer to this environment.

_unfold_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_unfold_python="${UNFOLD_PYTHON:-python3.11}"

if [ ! -x "${_unfold_root}/.venv/bin/python" ]; then
  if ! command -v "${_unfold_python}" >/dev/null 2>&1; then
    echo "env.sh: ${_unfold_python} not found; install it (brew install python@3.11) or set UNFOLD_PYTHON" >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "env.sh: creating ${_unfold_root}/.venv with ${_unfold_python}"
  "${_unfold_python}" -m venv "${_unfold_root}/.venv" || return 1
  "${_unfold_root}/.venv/bin/pip" install --upgrade pip -q
fi

# shellcheck disable=SC1091
source "${_unfold_root}/.venv/bin/activate"

if ! "${_unfold_root}/.venv/bin/python" -c "import unfold" >/dev/null 2>&1; then
  echo "env.sh: installing the unfold package (editable)"
  pip install -e "${_unfold_root}" -q || return 1
fi

# ROOT (TUnfold) from the external build; RooUnfold only if it has been built.
# shellcheck disable=SC1091
source "${_unfold_root}/setup_root.sh" || return 1
if [ -f "${HOME}/opt/RooUnfold/libRooUnfold.so" ] || [ -f "${HOME}/opt/RooUnfold/libRooUnfold.dylib" ] \
   || [ -n "${UNFOLD_ROOUNFOLD_LIB:-}" ]; then
  # shellcheck disable=SC1091
  source "${_unfold_root}/setup_roounfold.sh" >/dev/null 2>&1 || true
fi

unset _unfold_root _unfold_python
echo "unfold environment ready: $(python --version 2>&1), ROOT $(root-config --version 2>/dev/null || echo '?')"
