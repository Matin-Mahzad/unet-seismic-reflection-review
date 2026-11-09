import sys; sys.path.insert(0, '/home/claude/work/figs')
from common import *

fig = plt.figure(figsize=(7.1, 4.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-4, 62); ax.axis('off')

cols = [
    ('Processing', 'pre-stack gathers: shot, receiver, CMP', DOM['Processing'], 2,
     [('Random-noise attenuation', 'R'), ('Coherent noise and multiples', 'R'), ('Deblending', 'R'),
      ('Trace interpolation', 'R'), ('Bandwidth extension', 'R'), ('First-break picking', 'S'),
      ('Velocity picking', 'S'), ('Statics and NMO correction', 'M')]),
    ('Imaging & inversion', 'change of domain: data → image or model', DOM['Imaging & inversion'], 35.5,
     [('Migration deblurring (LSM)', 'M'), ('Image conditioning, diffractions', 'R'), ('Velocity-model building, FWI', 'M'),
      ('Low-frequency extrapolation', 'M'), ('Impedance and elastic inversion', 'M'), ('Petrophysical properties', 'M')]),
    ('Interpretation', 'migrated, stacked image volumes', DOM['Interpretation'], 69,
     [('Faults', 'S'), ('Horizons and relative geologic time', 'S'), ('Salt bodies', 'S'), ('Facies and stratigraphy', 'S'),
      ('Channels, karst, geobodies', 'S'), ('Attributes, DHIs, 4D and CO$_2$', 'S')]),
]
W = 29
vm, rec, tt, tfb = fd_data()
gather = gained(rec, tt)
seis, _, _ = section(nz=128, nx=160, seed=11, throw=10)
thumbs = [(gather, dict(cmap='gray_r', vmin=-0.6, vmax=0.6)), (vm, dict(cmap='turbo', vmin=1500, vmax=4600)), (seis, dict(cmap='gray_r', vmin=-1, vmax=1))]
for (title, sub, c, x0, tasks), (img, kw) in zip(cols, thumbs):
    box(ax, x0, 2.0, W, 58.5, fc=LGT, ec='#D5DBE3', lw=0.6, r=0.8, z=0)
    box(ax, x0, 55.0, W, 5.5, fc=c, r=0.8, z=1)
    ax.text(x0 + W / 2, 57.75, title, ha='center', va='center', color='white', fontsize=8.5, fontweight='bold', zorder=3)
    ax.text(x0 + W / 2, 53.3, sub, ha='center', va='center', color=INK, fontsize=6.4, style='italic')
    # thumbnail
    ia = fig.add_axes([(x0 + 4.5) / 100, (37.6 + 4) / 66, (W - 9) / 100, 13.6 / 66])
    ia.imshow(img, aspect='auto', interpolation='bilinear', **kw); ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values(): s.set_visible(True); s.set_linewidth(0.5); s.set_color('#9CA3AF')
    y = 36.0
    dy = 34.0 / max(len(tasks), 6)
    for name, role in tasks:
        box(ax, x0 + 1.2, y - dy + 0.55, W - 2.4, dy - 1.0, fc='white', ec='#D5DBE3', lw=0.5, r=0.5, z=1)
        ax.add_patch(plt.Circle((x0 + 3.4, y - dy / 2 + 0.05), 1.15, color=ROLE[role], zorder=3))
        ax.text(x0 + 3.4, y - dy / 2 + 0.05, role, ha='center', va='center', color='white', fontsize=6.2, fontweight='bold', zorder=4)
        ax.text(x0 + 5.6, y - dy / 2 + 0.05, name, ha='left', va='center', color=INK, fontsize=7, zorder=4)
        y -= dy
# flow arrows between columns
for xa in (31.2, 64.7):
    arrow(ax, xa, 45.5, xa + 4.1, 45.5, c=INK, lw=1.2, ms=9, z=5)
# legend
for i, (role, lab) in enumerate((('R', 'Restoration: data → cleaner data, same domain'), ('M', 'Mapping: data → another physical quantity'), ('S', 'Segmentation: data → labels or probabilities'))):
    xx = 2 + i * 33.5
    ax.add_patch(plt.Circle((xx + 1.2, -1.8), 1.1, color=ROLE[role], zorder=3))
    ax.text(xx + 1.2, -1.8, role, ha='center', va='center', color='white', fontsize=6, fontweight='bold', zorder=4)
    ax.text(xx + 3.2, -1.8, lab, ha='left', va='center', color=INK, fontsize=6.6)
save(fig, 'Fig01_workflow')
