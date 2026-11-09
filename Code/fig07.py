import sys; sys.path.insert(0, '/home/claude/work/figs')
from common import *

fig, axs = plt.subplots(2, 4, figsize=(7.1, 5.0))
plt.subplots_adjust(wspace=0.08, hspace=0.18, left=0.01, right=0.99, top=0.95, bottom=0.01)
for a in axs.flat:
    a.set_xlim(0, 10); a.set_ylim(0, 12); a.axis('off')

def B(a, x, y, w, h, t, c, tc='white', fs=6.2):
    box(a, x - w / 2, y - h / 2, w, h, fc=c, ec=c, r=0.25, text=t, fs=fs, tc=tc, z=3)
def A(a, x0, y0, x1, y1, c=GRY, **kw):
    arrow(a, x0, y0, x1, y1, c=c, lw=0.8, ms=5.5, z=2, **kw)
def op(a, x, y, s, c=INK):
    a.add_patch(plt.Circle((x, y), 0.42, fc='white', ec=c, lw=0.8, zorder=4))
    a.text(x, y, s, ha='center', va='center', fontsize=7, color=c, zorder=5)
def title(a, s, sub):
    a.text(0.2, 11.7, s, fontsize=7.8, fontweight='bold', color=INK, va='top')
    a.text(0.2, 10.85, sub, fontsize=6.2, color=GRY, va='top')
CONV, NORM, ACT = '#4D80C4', '#9DBCE3', '#6B7280'

# (a) plain block
a = axs[0, 0]; title(a, '(a) Plain block', 'U-Net (2015)')
ys = [9.2, 8.0, 6.8, 5.4, 4.2, 3.0]
labs = [('3×3 conv', CONV, 'white'), ('norm', NORM, INK), ('ReLU', ACT, 'white')] * 2
for y, (t, c, tc) in zip(ys, labs):
    B(a, 5, y, 4.2, 0.85, t, c, tc)
for y0, y1 in zip(ys[:-1], ys[1:]):
    A(a, 5, y0 - 0.43, 5, y1 + 0.43)
A(a, 5, 10.2, 5, 9.63); A(a, 5, 2.57, 5, 1.6)
a.text(5, 1.2, '$\\mathbf{h}_{out}$', ha='center', fontsize=7); a.text(5.3, 10.3, '$\\mathbf{h}_{in}$', fontsize=7)

# (b) residual
a = axs[0, 1]; title(a, '(b) Residual block', 'ResNet (2016), ResUNet (2018)')
ys = [8.6, 7.3, 6.0, 4.7]
for y, (t, c, tc) in zip(ys, [('3×3 conv', CONV, 'white'), ('norm + ReLU', NORM, INK), ('3×3 conv', CONV, 'white'), ('norm', NORM, INK)]):
    B(a, 4.2, y, 4.0, 0.85, t, c, tc)
for y0, y1 in zip(ys[:-1], ys[1:]): A(a, 4.2, y0 - 0.43, 4.2, y1 + 0.43)
A(a, 4.2, 9.9, 4.2, 9.03); A(a, 4.2, 4.27, 4.2, 3.0)
op(a, 4.2, 2.55, '+', NEW)
a.plot([4.2, 8.2, 8.2], [9.7, 9.7, 2.55], color=NEW, lw=1.0); A(a, 8.2, 2.55, 4.62, 2.55, c=NEW)
a.text(8.4, 6.2, 'identity', rotation=90, fontsize=6.2, color=NEW, va='center')
a.text(1.0, 6.6, '$\\mathcal{F}(\\mathbf{h})$', fontsize=7, color=INK)
A(a, 4.2, 2.13, 4.2, 1.3); a.text(4.5, 0.9, 'ReLU → $\\mathbf{h}+\\mathcal{F}(\\mathbf{h})$', fontsize=6.3)

# (c) dense
a = axs[0, 2]; title(a, '(c) Dense block', 'DenseNet (2017)')
xs = [1.6, 4.0, 6.4, 8.8]
for i, x in enumerate(xs):
    B(a, x, 5.5, 1.3, 2.6, f'$\\mathbf{{x}}_{i}$', ['#9DBCE3', '#6E9AD2', '#4D80C4', '#2F5A99'][i], 'white', 7)
for i in range(3):
    A(a, xs[i] + 0.65, 5.5, xs[i + 1] - 0.65, 5.5, c=INK)
    a.text((xs[i] + xs[i + 1]) / 2, 5.85, f'$H_{i+1}$', ha='center', fontsize=6.5)
for i in range(4):
    for j in range(i + 2, 4):
        rad = -0.35 - 0.1 * (j - i)
        A(a, xs[i], 6.85, xs[j] - 0.1, 6.85, c=NEW, cs=f'arc3,rad={rad}')
a.text(5, 2.6, '$\\mathbf{x}_l = H_l([\\mathbf{x}_0,\\ldots,\\mathbf{x}_{l-1}])$', ha='center', fontsize=7)
a.text(5, 1.6, 'every layer sees all earlier\nfeature maps (concatenation)', ha='center', fontsize=6.2, color=GRY, va='top')

# (d) squeeze-and-excitation
a = axs[0, 3]; title(a, '(d) Squeeze-and-excitation', 'SE-Net (2018); scSE, CBAM')
for k in range(4):
    box(a, 0.7 + 0.25 * k, 5.0 - 0.25 * k, 1.6, 3.2, fc=['#C9D9EF', '#9DBCE3', '#6E9AD2', '#4D80C4'][k], ec='white', lw=0.5, r=0.05, z=2 + k)
a.text(1.5, 2.7, '$\\mathbf{h}$ (C ch.)', ha='center', fontsize=6.3)
ys = [9.2, 8.0, 6.8, 5.6, 4.4]
for y, (t, c, tc) in zip(ys, [('global avg. pool', '#6B7280', 'white'), ('FC  C → C/r', CONV, 'white'), ('ReLU', NORM, INK), ('FC  C/r → C', CONV, 'white'), ('sigmoid → $\\mathbf{s}$', NEW, 'white')]):
    B(a, 6.4, y, 4.2, 0.85, t, c, tc)
for y0, y1 in zip(ys[:-1], ys[1:]): A(a, 6.4, y0 - 0.43, 6.4, y1 + 0.43)
A(a, 2.3, 8.3, 4.25, 9.2)
op(a, 6.4, 2.9, '×', NEW); A(a, 6.4, 3.97, 6.4, 3.32, c=NEW); A(a, 2.6, 3.6, 5.98, 2.9)
A(a, 6.4, 2.48, 6.4, 1.5); a.text(6.4, 1.05, '$\\tilde{\\mathbf{h}}_c = s_c\\,\\mathbf{h}_c$', ha='center', fontsize=7)

# (e) attention gate
a = axs[1, 0]; title(a, '(e) Attention gate', 'Attention U-Net (2018)')
B(a, 2.2, 9.0, 2.6, 0.9, '$\\mathbf{x}_l$ (skip)', '#4D80C4'); B(a, 7.5, 9.0, 3.0, 0.9, '$\\mathbf{g}$ (coarser)', DEC)
B(a, 2.2, 7.5, 2.4, 0.8, '1×1  $\\mathbf{W}_x$', NORM, INK); B(a, 7.6, 7.5, 2.4, 0.8, '1×1  $\\mathbf{W}_g$', NORM, INK)
A(a, 2.2, 8.55, 2.2, 7.9); A(a, 7.6, 8.55, 7.6, 7.9)
op(a, 4.9, 6.2, '+'); A(a, 2.2, 7.1, 4.55, 6.35); A(a, 7.6, 7.1, 5.25, 6.35)
ys = [5.1, 4.0, 2.9]
for y, (t, c, tc) in zip(ys, [('ReLU', ACT, 'white'), ('1×1  $\\boldsymbol{\\psi}$', NORM, INK), ('sigmoid → $\\alpha$', NEW, 'white')]):
    B(a, 4.9, y, 3.0, 0.8, t, c, tc)
A(a, 4.9, 5.78, 4.9, 5.5)
for y0, y1 in zip(ys[:-1], ys[1:]): A(a, 4.9, y0 - 0.4, 4.9, y1 + 0.4)
op(a, 1.4, 1.5, '×', NEW); A(a, 3.4, 2.9, 1.82, 1.6, c=NEW)
a.plot([0.7, 0.7], [9.0, 1.5], color='#4D80C4', lw=0.9); a.plot([0.7, 0.95], [9.0, 9.0], color='#4D80C4', lw=0.9)
A(a, 0.7, 1.5, 0.98, 1.5, c='#4D80C4')
A(a, 1.82, 1.5, 3.2, 1.5); a.text(3.35, 1.5, '$\\hat{\\mathbf{x}}_l$ → decoder', fontsize=6.5, va='center')

# (f) dilated / ASPP
a = axs[1, 1]; title(a, '(f) Dilated conv. and ASPP', 'dilated conv. (2016); DeepLab (2018)')
for i, r in enumerate([1, 2, 3]):
    cx = 1.9 + i * 3.1; cy = 7.2
    for gx in range(7):
        for gy in range(7):
            a.add_patch(Rectangle((cx - 1.4 + gx * 0.4, cy - 1.4 + gy * 0.4), 0.4, 0.4, fc='white', ec='#D5DBE3', lw=0.3, zorder=1))
    for kx in (-1, 0, 1):
        for ky in (-1, 0, 1):
            a.add_patch(Rectangle((cx - 0.2 + kx * r * 0.4, cy - 0.2 + ky * r * 0.4), 0.4, 0.4, fc=CONV, ec='white', lw=0.3, zorder=2))
    a.text(cx, 5.3, f'$r$ = {r}', ha='center', fontsize=6.5)
for i, r in enumerate(['1×1', '$r$=6', '$r$=12', 'pool']):
    B(a, 1.6 + i * 2.25, 3.4, 2.0, 0.8, r, [CONV, CONV, CONV, ACT][i])
    A(a, 1.6 + i * 2.25, 3.0, 5.0, 2.05, c=GRY)
B(a, 5.0, 1.6, 5.6, 0.8, 'concatenate → 1×1 conv', NEW)

# (g) UNet++ nested skips
a = axs[1, 2]; title(a, '(g) Nested skip pathways', 'UNet++ (2018), UNet 3+ (2020)')
pos = {}
for i in range(4):
    for j in range(4 - i):
        x = 1.4 + j * 2.4 + i * 0.0; y = 9.0 - i * 2.1
        pos[(i, j)] = (x, y)
        c = '#4D80C4' if j == 0 else (DEC if i + j == 3 else NEW)
        a.add_patch(plt.Circle((x, y), 0.5, fc=c, ec='white', lw=0.6, zorder=3))
        a.text(x, y, f'{i},{j}', ha='center', va='center', fontsize=5.5, color='white', zorder=4)
for i in range(3):
    A(a, pos[(i, 0)][0], pos[(i, 0)][1] - 0.5, pos[(i + 1, 0)][0], pos[(i + 1, 0)][1] + 0.5, c='#D1495B')
for (i, j), (x, y) in pos.items():
    if j > 0:
        xs_, ys_ = pos[(i + 1, j - 1)]; A(a, xs_ + 0.4, ys_ + 0.35, x - 0.35, y - 0.4, c=DEC)
        for k in range(j):
            xk, yk = pos[(i, k)]
            A(a, xk + 0.5, yk + (0.12 if k < j - 1 else 0), x - 0.5, y, c=GRY, cs='arc3,rad=-0.25' if k < j - 1 else 'arc3,rad=0')
for j in range(1, 4):
    x, y = pos[(0, j)]; a.text(x, y + 0.85, '$\\mathcal{L}$', ha='center', fontsize=7, color=NEW)
a.text(5.0, 1.3, 'dense skip pathways + deep supervision', ha='center', fontsize=6.3, color=GRY)

# (h) transformer block
a = axs[1, 3]; title(a, '(h) Transformer block', 'ViT (2021); TransUNet, Swin')
for k in range(6):
    box(a, 0.6 + k * 0.62, 9.0, 0.5, 0.5, fc=['#9DBCE3', '#6E9AD2'][k % 2], ec='white', r=0.05, z=3)
a.text(2.3, 8.5, 'tokens $\\mathbf{X}\\in\\mathbb{R}^{N\\times d}$', ha='center', fontsize=6.2)
ys = [7.3, 6.1, 4.5, 3.3]
for y, (t, c, tc) in zip(ys, [('LayerNorm', NORM, INK), ('multi-head attention', NEW, 'white'), ('LayerNorm', NORM, INK), ('MLP', CONV, 'white')]):
    B(a, 2.6, y, 4.3, 0.8, t, c, tc, fs=5.9)
A(a, 2.6, 8.15, 2.6, 7.7); A(a, 2.6, 6.9, 2.6, 6.5)
op(a, 2.6, 5.3, '+'); A(a, 2.6, 5.7, 2.6, 5.72); A(a, 2.6, 4.88, 2.6, 4.9)
A(a, 2.6, 3.7 + 0.2, 2.6, 3.7); op(a, 2.6, 2.1, '+'); A(a, 2.6, 2.9, 2.6, 2.52)
a.plot([0.2, 0.2], [7.9, 5.3], color=GRY, lw=0.7); A(a, 0.2, 5.3, 2.18, 5.3)
a.plot([0.2, 0.2], [4.95, 2.1], color=GRY, lw=0.7); A(a, 0.2, 2.1, 2.18, 2.1)
rng = np.random.default_rng(3)
M = rng.random((8, 8)) ** 3; M = M / M.sum(1, keepdims=True) + np.eye(8) * 0.4
ia = a.inset_axes([0.62, 0.3, 0.33, 0.27]); ia.imshow(M, cmap='Reds'); ia.set_xticks([]); ia.set_yticks([])
a.text(7.9, 7.2, 'softmax$(\\mathbf{QK}^\\top/\\sqrt{d_k})$', ha='center', fontsize=6.2)
a.text(7.9, 3.0, 'cost $\\mathcal{O}(N^2 d)$', ha='center', fontsize=6.3, color=NEW)
save(fig, 'Fig06_blocks')
