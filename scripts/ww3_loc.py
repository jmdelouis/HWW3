"""Size of the WW3 routines that an HWW3 run with the explicit scheme uses.

usage: python ww3_loc.py <WW3 model/src> <switch file>

For each group of routines (propagation, ST4, DIA, ...) prints the raw lines
between SUBROUTINE and END SUBROUTINE and the active code lines: the file is
preprocessed by cpp with -DW3_<switch> for every switch of the switch file,
then blank and comment-only lines are dropped (declarations and continuation
lines are counted). Writes ww3_loc.json in the current directory.
"""
import re, subprocess, sys, json
src = sys.argv[1]; switch = open(sys.argv[2]).read().split()
defs = ["-DW3_" + s for s in switch]
groups = {
 "geo_prop_scheme": {"w3profsmd_pdlib.F90": ["PDLIB_W3XYPUG", "PDLIB_W3XYPFSN2"]},
 "geo_prop_geometry": {"w3profsmd_pdlib.F90": ["PDLIB_INIT", "PDLIB_MAPSTA_INIT", "PDLIB_IOBP_INIT", "SET_IOBDP_PDLIB", "SET_IOBPA_PDLIB", "SET_UG_IOBP_PDLIB_INIT"],
                       "w3triamd.F90": ["SPATIAL_GRID", "NVECTRI", "COUNT", "COORDMAX", "AREA_SI", "SET_IOBP", "SET_UG_IOBP", "GET_BOUNDARY"]},
 "spectral_prop": {"w3pro3md.F90": ["W3KTP3"], "w3uqckmd.F90": ["W3QCK1", "W3QCK2"], "w3triamd.F90": ["UG_GRADIENTS"]},
 "st4": {"w3src4md.F90": ["W3SPR4", "W3SIN4", "CALC_USTAR", "W3SDS4"]},
 "st4_tables_init": {"w3src4md.F90": ["INSIN4", "TABU_STRESS", "TABU_TAUHF", "TABU_TAUHF2"]},
 "dia": {"w3snl1md.F90": ["W3SNL1", "INSNL1"]},
 "ln1": {"w3sln1md.F90": ["W3SLN1"]},
 "db1": {"w3sdb1md.F90": ["W3SDB1"]},
 "bt4": {"w3sbt4md.F90": ["INSBT4", "TABU_ERF", "W3SBT4"]},
 "source_integration": {"w3srcemd.F90": ["W3SRCE"]},
 "forcing_update": {"w3updtmd.F90": ["W3UWND", "W3UCUR", "W3ULEV", "W3UICE", "W3UBPT"], "w3iobcmd.F90": ["W3IOBC"]},
 "outputs": {"w3iogomd.F90": ["W3OUTG"]},
 "main_loop": {"w3wavemd.F90": ["W3WAVE"]},
}
def active(path):
    out = subprocess.run(["cpp", "-P", "-traditional-cpp", "-w"] + defs + [path], capture_output=True, text=True).stdout
    return out.splitlines()
def routines(lines):
    res, cur, start = {}, None, 0
    for i, l in enumerate(lines):
        s = l.strip().upper()
        m = re.match(r"^(RECURSIVE\s+|PURE\s+|ELEMENTAL\s+)*(REAL\s+|INTEGER\s+|LOGICAL\s+)?(SUBROUTINE|FUNCTION)\s+(\w+)", s)
        if m and cur is None:
            cur, start = m.group(4), i
        elif cur and re.match(r"^END\s+(SUBROUTINE|FUNCTION)", s):
            res.setdefault(cur, (start, i)); cur = None
    return res
def code(lines):
    return sum(1 for l in lines if l.strip() and not l.strip().startswith("!"))
out = {}
for g, files in groups.items():
    tot_raw = tot_act = 0; det = []
    for f, names in files.items():
        raw = open(f"{src}/{f}", errors="replace").read().splitlines()
        act = active(f"{src}/{f}")
        rr, ra = routines(raw), routines(act)
        for n in names:
            a, b = rr.get(n, (0, -1)); c, d = ra.get(n, (0, -1))
            nraw = b - a + 1; nact = code(act[c:d + 1]) if n in ra else 0
            tot_raw += nraw; tot_act += nact; det.append(f"{n} {nraw}/{nact}")
    out[g] = {"raw": tot_raw, "active_code": tot_act, "routines": det}
    print(f"{g:22s} raw {tot_raw:6d}  active code {tot_act:6d}   " + ", ".join(det))
json.dump(out, open("ww3_loc.json", "w"), indent=1)
