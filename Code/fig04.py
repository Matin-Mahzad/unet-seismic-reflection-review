import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

fig = plt.figure(figsize=(7.1, 3.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis('off')

levels = 4
H = [16, 12, 8.6, 6.2, 4.4]            # drawn heights (resolution)
Wd = [1.0, 1.3, 1.6, 1.9, 2.2]        # drawn widths (channels)
ytop = [40, 30, 22, 15.6, 10.2]       # y of block centre
xe = [15.5, 20.0, 25.0, 30.5]         # encoder x
xb = 36.8                              # bottleneck x
xd = [63.0, 57.3, 50.9, 43.8]         # decoder x (level 1..4)
shapes = ['H×W×C', 'H/2×W/2×2C', 'H/4×W/4×4C', 'H/8×W/8×8C', 'H/16×W/16×16C']
rf = [5, 14, 32, 68, 140]
encc = ['#9DBCE3', '#6E9AD2', '#4D80C4', '#3B6FB6', BOT]
decc = ['#9FD8CF', '#6EC1B5', '#45AA9C', '#2A9D8F']

def block(x, yc, h, w, c, double=True):
    g = 0.25
    box(ax, x, yc - h / 2, w, h, fc=c, ec='#33415580', lw=0.5, r=0.15, z=3)
    if double:
        box(ax, x + w + g, yc - h / 2, w, h, fc=c, ec='#33415580', lw=0.5, r=0.15, z=3)
    return x + (2 * w + g if double else w)

# input thumbnail
seis, imp, fm = section(nz=128, nx=96, seed=11, throw=8)
rng = np.random.default_rng(2)
noisy = seis + 0.5 * gaussian_filter(rng.normal(0, 1, seis.shape), 0.5) * 1.6
ia = fig.add_axes([0.012, (40 - 8) / 56, 0.105, 16 / 56]); ia.imshow(noisy, cmap='gray_r', vmin=-1.2, vmax=1.2, aspect='auto')
ia.set_xticks([]); ia.set_yticks([])
ax.text(6.4, 49.2, 'Input $\\mathbf{x}$', ha='center', fontsize=7.5, fontweight='bold', color=INK)
ax.text(6.4, 31.0, 'gather, section\nor volume patch', ha='center', va='top', fontsize=6.3, color=GRY)
arrow(ax, 11.6, 40, 15.2, 40, c=INK, lw=0.9)

ends_e = []
for l in range(levels):
    xr = block(xe[l], ytop[l], H[l], Wd[l], encc[l])
    ends_e.append(xr)
    ax.text(xe[l] - 0.4, ytop[l] + H[l] / 2 + 0.5, shapes[l], fontsize=5.4, color=INK, ha='left', va='bottom')
    ax.text(xe[l] + 0.2, ytop[l] - H[l] / 2 - 0.5, f'$r$={rf[l]}', fontsize=5.6, color=ENC, ha='left', va='top')
    # down arrow to next level
    nx = xe[l + 1] if l + 1 < levels else xb
    arrow(ax, xr - Wd[l], ytop[l] - H[l] / 2 - 0.2, nx + 0.4, ytop[l + 1] + H[l + 1] / 2 + 0.2, c='#D1495B', lw=1.0, ms=6, z=5)
# bottleneck
xr = block(xb, ytop[4], H[4], Wd[4], BOT)
ax.text(xb + 2.3, ytop[4] - H[4] / 2 - 0.6, 'bottleneck  ' + shapes[4] + f',  $r$={rf[4]}', fontsize=5.8, ha='center', va='top', color='#9A6A12')
# decoder
for l in range(levels - 1, -1, -1):
    x = xd[l]
    # up arrow from below
    src_x = xb + 2 * Wd[4] if l == levels - 1 else xd[l + 1] + 3 * Wd[l + 1]
    src_y = ytop[l + 1] + H[l + 1] / 2 + 0.2
    arrow(ax, src_x - 0.6, src_y, x + 0.8, ytop[l] - H[l] / 2 - 0.2, c=DEC, lw=1.1, ms=6, z=5)
    # concatenated pair: skip copy (encoder colour) + upsampled (decoder colour), then conv block
    box(ax, x, ytop[l] - H[l] / 2, Wd[l] * 0.55, H[l], fc=encc[l], ec='#33415580', lw=0.5, r=0.1, z=3)
    box(ax, x + Wd[l] * 0.55, ytop[l] - H[l] / 2, Wd[l] * 0.55, H[l], fc=decc[l], ec='#33415580', lw=0.5, r=0.1, z=3)
    block(x + Wd[l] * 1.1 + 0.25, ytop[l], H[l], Wd[l], decc[l])
    # skip
    arrow(ax, ends_e[l] + 0.3, ytop[l] + H[l] * 0.18, x - 0.2, ytop[l] + H[l] * 0.18, c=GRY, lw=0.8, ms=6, ls=(0, (3, 2)))
ax.text(40.0, ytop[0] + H[0] * 0.18 + 1.0, 'skip connection: copy and concatenate  $[\\,\\mathrm{U}(\\mathbf{u}_{l+1})\\,\\Vert\\,\\mathbf{e}_l\\,]$', ha='center', fontsize=6.4, style='italic', color=GRY)
ax.text(23.0, 53.6, 'ENCODER  (contracting)', ha='center', fontsize=7.5, fontweight='bold', color=ENC)
ax.text(23.0, 51.7, 'context ↑   resolution ↓', ha='center', fontsize=6.3, color=GRY)
ax.text(57.0, 53.6, 'DECODER  (expanding)', ha='center', fontsize=7.5, fontweight='bold', color=DEC)
ax.text(57.0, 51.7, 'resolution ↑   localisation ↑', ha='center', fontsize=6.3, color=GRY)
# 1x1 conv and heads
xo = xd[0] + Wd[0] * 1.1 + 0.25 + 2 * Wd[0] + 0.25
box(ax, xo + 1.0, 37.6, 4.0, 4.8, fc='white', ec=INK, lw=0.7, r=0.3, text='1×1\nconv', fs=6.3, tc=INK, z=3)
arrow(ax, xo, 40, xo + 1.0, 40, c=INK, lw=0.8, ms=6)
heads = [('Restoration', 'identity / linear', '$\\hat{\\mathbf{y}}$ = clean data', ROLE['R'], 47.0, seis, dict(cmap='gray_r', vmin=-1.2, vmax=1.2)),
         ('Mapping', 'linear (scaled)', '$\\hat{\\mathbf{y}}$ = impedance, velocity', ROLE['M'], 30.0, np.log(imp), dict(cmap='viridis')),
         ('Segmentation', 'sigmoid / softmax', '$\\hat{\\mathbf{y}}$ = probability map', ROLE['S'], 13.0, gaussian_filter(fm, 0.8), dict(cmap='Purples'))]
for name, act, desc, c, yc, img, kw in heads:
    arrow(ax, xo + 5.0, 40, 76.0, yc, c=c, lw=0.9, ms=6, cs='arc3,rad=0.0')
    a = fig.add_axes([0.76, (yc - 6.5) / 56, 0.075, 13 / 56]); a.imshow(img, aspect='auto', **kw); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(True); s.set_color(c); s.set_linewidth(1.2)
    ax.text(84.2, yc + 3.5, name, fontsize=7.3, fontweight='bold', color=c, va='center')
    ax.text(84.2, yc + 0.8, act, fontsize=6.2, color=INK, va='center')
    ax.text(84.2, yc - 1.9, desc, fontsize=6.2, color=GRY, va='center')
# legend
ly = 2.2
items = [('#4D80C4', 'block $\\Phi_l$: 2 × (3×3 conv + norm + ReLU)'), ('#D1495B', '2×2 max-pool $\\mathrm{P}$'),
         (DEC, 'up-sampling $\\mathrm{U}$ (transposed conv or interpolation)'), (GRY, 'skip (concatenate)')]
xx = 3
for c, t in items:
    if c in ('#D1495B', DEC, GRY):
        arrow(ax, xx, ly, xx + 3.2, ly, c=c, lw=1.0, ms=6, ls=(0, (3, 2)) if c == GRY else '-')
    else:
        box(ax, xx, ly - 1.0, 1.0, 2.0, fc=c, r=0.1); box(ax, xx + 1.25, ly - 1.0, 1.0, 2.0, fc=c, r=0.1)
    ax.text(xx + 4.0, ly, t, fontsize=6.2, va='center', color=INK)
    xx += {0: 33, 1: 15, 2: 31, 3: 0}[items.index((c, t))]
save(fig, 'Figure_4')
