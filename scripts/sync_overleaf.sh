#!/bin/bash
# Mirror tex/hww3 (main.tex, *.bib, fig/*.png) to the Overleaf project.
# Needs an Overleaf git token in the git credential store (user "git").
# Usage: bash scripts/sync_overleaf.sh   (from the repository root)
set -e
PROJECT=${OVERLEAF_PROJECT:-6af6a96c4ed66a0cad905cb6}
SRC=$(cd "$(dirname "$0")/../tex/hww3" && pwd)
W=${OVERLEAF_DIR:-$HOME/overleaf_hww3}
if [ ! -d "$W/.git" ]; then
  GIT_TERMINAL_PROMPT=0 git -c credential.helper=store clone -q \
    "https://git.overleaf.com/$PROJECT" "$W"
  git -C "$W" config credential.helper store
fi
cd "$W"
GIT_TERMINAL_PROMPT=0 git pull -q origin main
cp "$SRC/main.tex" .
cp "$SRC"/*.bib . 2>/dev/null || true
mkdir -p fig && cp "$SRC"/fig/*.png fig/
git add -A
if git diff --cached --quiet; then echo "Overleaf already up to date"; exit 0; fi
git commit -q -m "Sync from HWW3 $(git -C "$SRC" rev-parse --short HEAD)"
GIT_TERMINAL_PROMPT=0 git push -q origin HEAD:main
git log --oneline | head -1
