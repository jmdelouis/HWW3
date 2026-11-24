# Glossary

The words used across this site, in alphabetical order.

Altimeter (L3)
: A satellite instrument that measures $H_s$ along the satellite's track.
  The L3 product of Copernicus Marine (`WAVE_GLO_PHY_SWH_L3_NRT_014_001`)
  gives one sample per second, about 7 km apart; HWW3 averages 7
  consecutive samples into a *block* of about 50 km.

`BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM`
: The four ST4 parameters of the calibration: the growth of the waves by
  the wind (`BETAMAX`, 1.75 in the reference runs), the whitecapping
  dissipation (`SDSC2`, −2.2e-5), the strength of the nonlinear
  interactions of the DIA (`NLPROP`, 2.5e7) and the breaking in shallow
  water (`BJGAM`, 0.73).

Buoy (moored)
: A floating instrument at a fixed position that measures $H_s$ and
  $T_{m02}$, among others. HWW3 uses the January 2026 files of the
  Copernicus Marine in situ product
  `INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030`, averaged per hour.

Cost ($J$)
: The number the calibration minimises: the sum of the squared misfits
  between model and observations, each divided by its expected error,
  plus a term that keeps the parameters near their start values.

Datarmor, hpc cluster
: The two Ifremer computers used. Datarmor runs PBS jobs (`qsub`) and
  holds the data mirrors; the hpc cluster runs Slurm jobs (`sbatch`) and
  has the A100 GPUs.

Explicit, implicit scheme
: Two ways WW3 advances the waves from node to node. The implicit scheme
  solves a linear system and allows long time steps; the explicit scheme
  is simpler but needs sub-steps where the cells are small. The GPU model
  rewrites the explicit scheme.

Forcing
: The fields that drive the model: the 10 m wind (ERA5), the surface
  currents and the water level (Copernicus Marine, with the FES2014 tide),
  and the sea-ice concentration (ERA5).

Gauss–Newton step
: One step of the calibration: from the misfits and their derivatives
  with respect to the parameters, the change of the parameters that would
  reduce the cost most if the model were linear. A value run then checks
  that the cost really decreased.

GEBCO
: The global bathymetry (ocean depth) grid, 15 arc-seconds (about 460 m),
  from which the grid generator takes its depths.

gridlook
: A web viewer of HEALPix maps, which draws the HWW3 outputs without
  interpolation.

HEALPix, nside
: A division of the sphere into cells of equal area. Each level splits a
  cell into four; `nside` = $2^{\text{level}}$ sets the cell size: 51 km at
  nside 128, 3.2 km at nside 2048. Each node of an HWW3 grid is the centre
  of a HEALPix cell.

$H_s$ (significant wave height)
: Four times the square root of the total energy of the spectrum (the
  variance of the sea-surface elevation): close to the mean height of the
  highest third of the waves, in metres.

`mod_def.ww3`
: The binary file in which `ww3_grid` stores the mesh, the spectral grid
  and the model parameters; every other WW3 program reads it.

Open boundary, boundary spectra
: The nodes on the edge of the domain where waves come in from outside.
  Their spectra come from a global run (Ifremer `GLOBMULTI`) and are
  written by `ww3_bounc` to `nest.ww3`.

Restart
: A file holding the full model state (the spectrum at every node) at one
  time, from which another run starts.

Spectrum
: The wave energy at one node per frequency and direction: 32 frequencies
  from 0.0373 Hz and 36 directions in HWW3.

ST4
: The set of source terms used by HWW3 (wind input, whitecapping, swell
  dissipation), with the DIA nonlinear interactions, shallow-water breaking
  and bottom friction.

Tangent run
: A run of the GPU model that computes, along with the fields, their
  derivative with respect to one parameter (or the wind, or the initial
  state). One tangent run per parameter gives the derivatives that the
  calibration needs, without finite differences.

$T_{m02}$ (mean period)
: A mean wave period computed from the spectrum, in seconds.

WW3 programs
: `ww3_grid` (grid and parameters to `mod_def.ww3`), `ww3_prnc` (forcing
  onto the nodes), `ww3_bounc` (boundary spectra), `ww3_strt` (initial
  state), `ww3_shel` (the model run) and `ww3_ounf` (outputs to netCDF,
  `ww3.202601.nc`).
