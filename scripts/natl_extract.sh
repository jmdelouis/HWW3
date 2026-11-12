#!/bin/bash
# Plotting subsets of the North Atlantic grid (run from /work/jmdeloui/HWW3).
PY=/work/jmdeloui/highway/venv/bin/python
D=grids/natl
mkdir -p $D/plot
$PY scripts/extract_zoom.py $D $D/plot/nodes.npz --nodes
$PY scripts/extract_zoom.py $D $D/plot/brittany.npz -6.5 -1.5 46.5 49.5
$PY scripts/extract_zoom.py $D $D/plot/iroise.npz -5.4 -4.2 47.9 48.7
$PY scripts/extract_zoom.py $D $D/plot/gibraltar.npz -7 -4 35 37
$PY scripts/extract_zoom.py $D $D/plot/openocean.npz -32 -24 38 44
ls -la $D/plot
