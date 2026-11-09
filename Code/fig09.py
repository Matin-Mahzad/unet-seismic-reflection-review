import sys; sys.path.insert(0, '/home/claude/work/figs')
from common import *

fig = plt.figure(figsize=(7.1, 4.6))
gs = fig.add_gridspec(2, 6, hspace=0.12, wspace=0.1, left=0.01, right=0.99, top=0.97, bottom=0.02)
axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
for a in axes:
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis('off')
    box(a, 0.05, 0.1, 9.9, 9.8, fc=LGT, ec='#D5DBE3', lw=0.5, r=0.3, z=0)

def unet(a, x, y, s=1.0, lab='U-Net', c=ENC):
    pts = [(x - 1.1 * s, y + 0.8 * s), (x - 0.65 * s, y + 0.8 * s), (x - 0.25 * s, y - 0.25 * s), (x + 0.25 * s, y - 0.25 * s),
           (x + 0.65 * s, y + 0.8 * s), (x + 1.1 * s, y + 0.8 * s), (x + 0.5 * s, y - 0.8 * s), (x - 0.5 * s, y - 0.8 * s)]
    a.add_patch(Polygon(pts, closed=True, fc=c, ec='white', lw=0.6, zorder=3))
    a.text(x, y - 1.15 * s, lab, ha='center', va='top', fontsize=6.3, color=c, fontweight='bold')
def B(a, x, y, w, h, t, c='white', tc=INK, ec=GRY, fs=6.2):
    box(a, x - w / 2, y - h / 2, w, h, fc=c, ec=ec, lw=0.6, r=0.2, text=t, fs=fs, tc=tc, z=3)
def A(a, x0, y0, x1, y1, c=GRY, **kw):
    arrow(a, x0, y0, x1, y1, c=c, lw=0.8, ms=5.5, z=4, **kw)
def T(a, s, sub):
    a.text(0.4, 9.55, s, fontsize=7.8, fontweight='bold', va='top', color=INK)
    a.text(0.4, 8.75, sub, fontsize=6.1, va='top', color=GRY)

# (a) supervised
a = axes[0]; T(a, '(a) Supervised', 'synthetic or labelled pairs')
B(a, 2.2, 6.3, 3.4, 1.1, 'generator /\nlabelled data', c='white')
A(a, 2.2, 5.75, 2.2, 4.7); a.text(2.4, 5.2, '$(\\mathbf{x}, \\mathbf{y})$', fontsize=6.8)
unet(a, 2.2, 3.6, 0.9, '$f_\\theta$')
A(a, 3.3, 3.6, 5.2, 3.6); B(a, 6.0, 3.6, 1.5, 0.9, '$\\hat{\\mathbf{y}}$')
A(a, 6.75, 3.6, 7.6, 3.6); B(a, 8.6, 3.6, 1.8, 1.0, '$\\mathcal{L}(\\hat{\\mathbf{y}}, \\mathbf{y})$', c='#FBE3EC', ec=NEW)
a.plot([2.2, 8.6], [6.3 + 0.0, 6.3], alpha=0)
A(a, 3.9, 6.3, 8.6, 4.1, c=GRY, cs='arc3,rad=-0.25', ls=(0, (2, 2)))
a.text(6.6, 6.0, '$\\mathbf{y}$', fontsize=6.8, color=GRY)
a.text(5.0, 1.0, 'faults, salt, VMB, impedance', ha='center', fontsize=6.2, color=GRY)

# (b) blind spot
a = axes[1]; T(a, '(b) Self-supervised (blind-spot)', 'field data only, Eq. (13)')
rng = np.random.default_rng(0)
for i in range(5):
    for j in range(5):
        m = (i, j) in [(1, 3), (3, 1), (2, 2), (4, 4), (0, 0)]
        a.add_patch(Rectangle((0.8 + j * 0.5, 5.2 + i * 0.5), 0.5, 0.5, fc=NEW if m else ['#9DBCE3', '#C9D9EF'][(i + j) % 2], ec='white', lw=0.4, zorder=3))
a.text(2.05, 4.75, 'masked noisy $\\mathbf{x}_{\\setminus\\mathcal{M}}$', ha='center', fontsize=6.1)
A(a, 3.4, 6.45, 4.4, 6.45); unet(a, 5.5, 6.4, 0.9, '$f_\\theta$')
A(a, 6.6, 6.45, 7.4, 6.45); B(a, 8.4, 6.45, 2.1, 0.9, 'predict $\\mathcal{M}$')
B(a, 6.3, 2.6, 6.6, 1.1, '$\\sum_{j\\in\\mathcal{M}}(f_\\theta(\\cdot)_j - x_j)^2$', c='#FBE3EC', ec=NEW)
A(a, 8.4, 6.0, 8.4, 3.15)
a.text(5.0, 1.0, 'random noise: denoising, interpolation', ha='center', fontsize=6.2, color=GRY)

# (c) closed loop
a = axes[2]; T(a, '(c) Semi-supervised closed loop', 'wells + unlabelled traces, Eq. (15)')
B(a, 1.5, 6.0, 1.6, 0.9, '$\\mathbf{d}$')
A(a, 2.3, 6.0, 3.0, 6.0); unet(a, 4.0, 6.0, 0.8, 'inverse $f_\\phi$')
A(a, 5.0, 6.0, 5.8, 6.0); B(a, 6.6, 6.0, 1.6, 0.9, '$\\hat{\\mathbf{m}}$')
A(a, 7.4, 6.0, 7.9, 6.0); unet(a, 8.8, 6.0, 0.75, 'forward $g_\\psi$', c=DEC)
A(a, 8.8, 4.7, 8.8, 3.9, c=DEC); B(a, 8.4, 3.35, 2.4, 0.9, '$g_\\psi(\\hat{\\mathbf{m}}) \\approx \\mathbf{d}$', c='#FBE3EC', ec=NEW)
A(a, 6.6, 5.55, 4.0, 3.9, c=GRY); B(a, 3.4, 3.35, 3.4, 0.9, '$\\hat{\\mathbf{m}} \\approx \\mathbf{m}$ at wells', c='#FBE3EC', ec=NEW)
a.text(5.0, 1.0, 'impedance and reservoir properties', ha='center', fontsize=6.2, color=GRY)

# (d) physics coupled
a = axes[3]; T(a, '(d) Physics-coupled reparametrisation', 'network output passes through the wave equation, Eq. (14)')
B(a, 1.2, 5.0, 1.2, 0.9, '$\\mathbf{z}$')
A(a, 1.8, 5.0, 2.6, 5.0); unet(a, 3.6, 5.0, 0.85, '$g_\\theta$')
A(a, 4.6, 5.0, 5.3, 5.0)
vm = fd_data()[0]
ia = a.inset_axes([0.535, 0.37, 0.11, 0.28]); ia.imshow(vm, cmap='turbo', aspect='auto', vmin=1500, vmax=4600); ia.set_xticks([]); ia.set_yticks([])
a.text(5.95, 3.35, '$\\mathbf{m}$', ha='center', fontsize=6.8)
A(a, 6.6, 5.0, 7.2, 5.0); B(a, 7.9, 5.0, 1.3, 1.0, '$F$', c=GRY, tc='white', ec=GRY, fs=7.5)
a.text(7.9, 3.9, 'wave-equation\nmodelling', ha='center', va='top', fontsize=5.8, color=GRY)
A(a, 8.55, 5.0, 9.2, 5.0)
B(a, 8.0, 7.3, 3.4, 0.9, '$\\Vert F(g_\\theta(\\mathbf{z})) - \\mathbf{d}_{obs}\\Vert^2$', c='#FBE3EC', ec=NEW)
A(a, 9.4, 5.4, 9.2, 6.85, c=NEW)
A(a, 6.3, 7.3, 3.6, 6.0, c=NEW, ls=(0, (2, 2)), cs='arc3,rad=0.2'); a.text(4.3, 7.4, 'gradient to $\\theta$', fontsize=6, color=NEW)
a.text(5.0, 1.0, 'full-waveform inversion, joint inversion', ha='center', fontsize=6.2, color=GRY)

# (e) diffusion
a = axes[4]; T(a, '(e) Conditional diffusion', 'U-Net predicts the added noise, Eq. (12)')
xs = [1.2, 3.0, 4.8, 8.6]
noise = np.random.default_rng(1).normal(size=(40, 40))
seis, _, _ = section(nz=40, nx=40, seed=11, throw=4)
for i, (x, alpha) in enumerate(zip([1.2, 3.0, 8.6], [0.0, 0.5, 1.0])):
    img = alpha * seis + (1 - alpha) * noise * 0.6
    ia = a.inset_axes([(x - 0.7) / 10, 0.42, 0.14, 0.24]); ia.imshow(img, cmap='gray_r', vmin=-1.2, vmax=1.2, aspect='auto'); ia.set_xticks([]); ia.set_yticks([])
for x, lab in zip([1.2, 3.0, 8.6], ['$\\mathbf{x}_T$', '$\\mathbf{x}_t$', '$\\mathbf{x}_0$']):
    a.text(x, 3.7, lab, ha='center', fontsize=6.8)
A(a, 2.0, 5.4, 2.25, 5.4); A(a, 3.75, 5.4, 4.6, 5.4)
unet(a, 5.7, 5.4, 0.85, '$\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, \\mathbf{c})$', c=NEW)
A(a, 6.8, 5.4, 7.8, 5.4); a.text(7.3, 5.75, '…', ha='center', fontsize=8)
B(a, 5.7, 7.7, 3.2, 0.8, 'condition $\\mathbf{c}$: degraded data', c='white')
A(a, 5.7, 7.3, 5.7, 6.25)
a.text(5.0, 1.0, 'repeated sampling gives an ensemble of restorations', ha='center', fontsize=6.2, color=GRY)
save(fig, 'Fig08_paradigms')
