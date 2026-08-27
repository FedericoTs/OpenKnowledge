#!/bin/sh
# OpenKnowledge installer.
#
#   curl -fsSL https://raw.githubusercontent.com/FedericoTs/OpenKnowledge/main/install.sh | sh
#
# What it does, in order: check for git and a Python 3.11+, clone (or update)
# the repository into ~/Documents/Projects/OpenKnowledge, build a virtualenv
# inside that folder, install the package into it, and prove it works by
# running the audit command against the repository's own test corpus.
#
# What it does not do: touch anything outside that one folder. No sudo, no
# system packages, no ~/.bashrc edits, no PATH changes, nothing installed
# globally. Deleting the folder uninstalls it completely.
#
# Set OK_DIR to install somewhere else. Re-running is safe: an existing
# checkout is updated, and local changes stop it rather than being discarded.

set -eu

REPO="${OK_REPO:-https://github.com/FedericoTs/OpenKnowledge.git}"
DIR="${OK_DIR:-$HOME/Documents/Projects/OpenKnowledge}"
BRANCH="${OK_BRANCH:-main}"

say()  { printf '\033[36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[33m !\033[0m %s\n' "$1" >&2; }
die()  { printf '\033[31m !!\033[0m %s\n' "$1" >&2; exit 1; }

# --- what we need ------------------------------------------------------------

command -v git >/dev/null 2>&1 || die "git is not installed. Install it and re-run."

PY=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
[ -n "$PY" ] || die "no Python 3.11 or newer found. Install one from python.org and re-run."
say "using $($PY -V) at $(command -v "$PY")"

# --- get the code ------------------------------------------------------------

if [ -d "$DIR/.git" ]; then
    say "updating $DIR"
    # Never discard someone's work to make an install succeed.
    if [ -n "$(git -C "$DIR" status --porcelain)" ]; then
        warn "$DIR has uncommitted changes; leaving the code as it is"
    else
        git -C "$DIR" fetch --quiet origin "$BRANCH"
        git -C "$DIR" checkout --quiet "$BRANCH"
        git -C "$DIR" merge --quiet --ff-only "origin/$BRANCH" \
            || warn "could not fast-forward $BRANCH; leaving the code as it is"
    fi
else
    say "cloning into $DIR"
    mkdir -p "$(dirname "$DIR")"
    git clone --quiet --branch "$BRANCH" "$REPO" "$DIR"
fi

cd "$DIR"

# --- build the environment ---------------------------------------------------

say "building the environment (a minute or two the first time)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e .

[ -f .env ] || { cp .env.example .env; say "wrote .env - every setting in it is optional"; }
mkdir -p documents data

# --- prove it works ----------------------------------------------------------

say "checking it works"
.venv/bin/openknowledge audit evals/corpus/aveline --exit-zero >/dev/null \
    || die "the install completed but the audit command failed. Please open an issue."

cat <<EOF

OpenKnowledge is installed in $DIR

  Put it on your PATH for this shell:
      export PATH="$DIR/.venv/bin:\$PATH"

  Then, in order:
      openknowledge audit ~/policies     # free, offline, writes nothing
      openknowledge model list           # what your machine can run
      openknowledge serve                # http://localhost:8080

  The audit tier needs no model, no key and no GPU. Answering questions needs
  a model: 'openknowledge model use qwen3:8b' once Ollama is running.

EOF
