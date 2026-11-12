import sys, netCDF4, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.tri as mtri
d = netCDF4.Dataset(sys.argv[1])
lon = d['longitude'][:]; lat = d['latitude'][:]
tri = d['tri'][:] - 1
hs = d['hs'][12]; dr = d['dir'][12]
print('dir median', np.median(dr), 'uwnd', float(d['uwnd'][12].mean()), 'vwnd', float(d['vwnd'][12].mean()))
T = mtri.Triangulation(lon, lat, tri)
fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
for a, ext in zip(ax, [None, (-6, 8, 55, 63)]):
    p = a.tripcolor(T, hs, shading='gouraud', cmap='viridis', vmin=0, vmax=3.9)
    if ext:
        a.set_xlim(*ext[:2]); a.set_ylim(*ext[2:])
    a.set_aspect(1 / np.cos(np.radians(45 if ext is None else 59)))
    a.set_xlabel('longitude'); a.set_ylabel('latitude')
fig.colorbar(p, ax=ax, label='Hs (m)', shrink=0.8)
ax[0].set_title('Hs after 12 h, uniform 15 m/s westerly, cold start')
ax[1].set_title('North Sea / Norway')
fig.savefig(sys.argv[2], dpi=130, bbox_inches='tight')
