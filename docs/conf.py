"""Sphinx configuration of the HWW3 documentation."""

import datetime as dt
import pathlib
import shutil

project = "HWW3"
author = "Jean-Marc Delouis and HWW3 developers"
copyright = f"2026-{dt.date.today().year}, {author}"
root_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_design",
]
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

exclude_patterns = ["_build", "jupyter_execute", ".ipynb_checkpoints"]
templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_title = "HWW3"
html_static_path = ["_static"]
# The top bar shows the seven sections of the root toctree (index.md):
# Getting started, Preparing a run, Running the model, Learning the
# parameters, Analysis tools, Results, Reference. Every page belongs to one
# section and appears in the left sidebar of that section, so nothing is
# hidden in a "More" dropdown; keep the number of sections at or below
# header_links_before_dropdown.
html_theme_options = {
    "github_url": "https://github.com/jmdelouis/HWW3",
    "show_toc_level": 2,
    "header_links_before_dropdown": 8,
    "navbar_align": "left",
    # the search icon instead of the wide search field, to leave the room
    # of the top bar to the section names
    "navbar_persistent": ["search-button"],
    "show_nav_level": 1,
    "navigation_with_keys": False,
}
# The start page has no section of its own: no left sidebar there.
html_sidebars = {"index": []}

# The figures live once, in tex/hww3/fig (shared with the LaTeX note);
# copy them next to the pages at build time.
_here = pathlib.Path(__file__).parent
_src = _here.parent / "tex" / "hww3" / "fig"
_dst = _here / "_static" / "fig"
if _src.is_dir():
    _dst.mkdir(parents=True, exist_ok=True)
    for f in _src.glob("*.png"):
        shutil.copy2(f, _dst / f.name)
