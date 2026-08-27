#!/bin/sh
# OpenKnowledge installer.
#
#   curl -fsSL https://raw.githubusercontent.com/FedericoTs/OpenKnowledge/HEAD/install.sh | sh
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
# Empty means "whatever the remote calls its default branch". Naming one here
# is how the published install command 404'd: it said main, and there is no
# main - the repository's default is the branch it was created with.
BRANCH="${OK_BRANCH:-}"

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
        here="${BRANCH:-$(git -C "$DIR" rev-parse --abbrev-ref HEAD)}"
        git -C "$DIR" fetch --quiet origin "$here"
        git -C "$DIR" checkout --quiet "$here"
        git -C "$DIR" merge --quiet --ff-only "origin/$here" \
            || warn "could not fast-forward $here; leaving the code as it is"
    fi
elif [ -d "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
    # A folder that is neither empty nor a checkout. git would print its own
    # blunt fatal here; say what to do about it instead, and never delete it.
    die "$DIR already exists and is not an OpenKnowledge checkout.
     Move it aside, empty it, or install somewhere else:
         OK_DIR=\"\$HOME/Documents/Projects/OpenKnowledge2\" sh install.sh"
else
    say "cloning into $DIR"
    mkdir -p "$(dirname "$DIR")"
    if [ -n "$BRANCH" ]; then
        git clone --quiet --branch "$BRANCH" "$REPO" "$DIR"
    else
        git clone --quiet "$REPO" "$DIR"
    fi
fi

cd "$DIR"

# --- build the environment ---------------------------------------------------

say "building the environment (a minute or two the first time)"
[ -d .venv ] || "$PY" -m venv .venv

# Windows venvs put the interpreter in Scripts/, POSIX ones in bin/. Git Bash
# runs this script happily and then finds neither, if you only look in one.
if   [ -x .venv/bin/python ];        then BIN=".venv/bin"
elif [ -x .venv/Scripts/python.exe ]; then BIN=".venv/Scripts"
else die "the virtualenv has no interpreter in .venv/bin or .venv/Scripts"
fi

"$BIN/python" -m pip install --quiet --upgrade pip
"$BIN/python" -m pip install --quiet -e .

[ -f .env ] || { cp .env.example .env; say "wrote .env - every setting in it is optional"; }
mkdir -p documents data

# --- prove it works ----------------------------------------------------------

say "checking it works"
"$BIN/python" -m openknowledge.cli audit evals/corpus/aveline --exit-zero >/dev/null \
    || die "the install completed but the audit command failed. Please open an issue."

cat <<EOF

OpenKnowledge is installed in $DIR

  Put it on your PATH for this shell:
      export PATH="$DIR/$BIN:\$PATH"

  Or for every shell from now on - this script deliberately does not edit
  your shell profile, so run it yourself if you want that:
      echo 'export PATH="$DIR/$BIN:\$PATH"' >> ~/.bashrc

  Then, in order:
      openknowledge audit ~/policies     # free, offline, writes nothing
      openknowledge model list           # what your machine can run
      openknowledge serve                # http://localhost:8080

  The audit tier needs no model, no key and no GPU. Answering questions needs
  a model: 'openknowledge model use qwen3:8b' once Ollama is running.

EOF
