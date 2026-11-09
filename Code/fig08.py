import sys; sys.path.insert(0, '/home/claude/work/figs')
from common import *

TC = {'C': '#4D80C4', 'D': DEC, 'T': '#7B4FA0', 'W': '#A98BC9', 'G': NEW, 'B': BOT}
fig = plt.figure(figsize=(7.1, 5.3))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.28, wspace=0.06, left=0.02, right=0.98, top=0.95, bottom=0.08)

def U(ax, enc, bott, dec, title, sub, mark=None):
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis('off')
    ax.text(0.1, 10.9, title, fontsize=7.8, fontweight='bold', va='top', color=INK)
    ax.text(0.1, 10.1, sub, fontsize=6.2, va='top', color=GRY)
    hs = [3.0, 2.3, 1.7, 1.2]; ys = [7.3, 5.5, 4.0, 2.8]
    for l in range(4):
        xe = 0.6 + l * 1.05; xd = 9.4 - l * 1.05 - 0.75
        for x, t in ((xe, enc[l]), (xd, dec[l])):
            box(ax, x, ys[l] - hs[l] / 2, 0.75, hs[l], fc=TC[t], ec='white', lw=0.4, r=0.1, z=3)
            if t in 'TWG':
                ax.text(x + 0.375, ys[l], {'T': 'T', 'W': 'W', 'G': 'G'}[t], ha='center', va='center', color='white', fontsize=6, fontweight='bold', zorder=4)
        arrow(ax, xe + 0.75, ys[l], xd, ys[l], c=GRY, lw=0.6, ms=4, ls=(0, (2, 2)))
        if l < 3:
            arrow(ax, xe + 0.37, ys[l] - hs[l] / 2, xe + 1.05 + 0.2, ys[l + 1] + hs[l + 1] / 2, c='#D1495B', lw=0.7, ms=4)
            arrow(ax, xd - 1.05 + 0.55, ys[l + 1] + hs[l + 1] / 2, xd + 0.37, ys[l] - hs[l] / 2, c=DEC, lw=0.7, ms=4)
    # bottleneck
    for k, t in enumerate(bott):
        box(ax, 3.2, 0.9, 3.6, 0.95, fc=TC[t], ec='white', lw=0.4, r=0.1, z=3)
        ax.text(5.0, 1.375, {'T': 'Transformer ×L', 'B': 'conv', 'W': 'Swin blocks', 'G': 'global attention'}[t], ha='center', va='center', color='white', fontsize=5.8, zorder=4)
    arrow(ax, 0.6 + 3 * 1.05 + 0.37, ys[3] - hs[3] / 2, 3.9, 1.85, c='#D1495B', lw=0.7, ms=4)
    arrow(ax, 6.1, 1.85, 9.4 - 3 * 1.05 - 0.4, ys[3] - hs[3] / 2, c=DEC, lw=0.7, ms=4)

ax = fig.add_subplot(gs[0, 0]); U(ax, 'CCCC', 'T', 'DDDD', '(a) Transformer bottleneck', 'TransUNet (2021)')
ax = fig.add_subplot(gs[0, 2]); U(ax, 'WWWW', 'W', 'WWWW', '(c) Window attention throughout', 'Swin-Unet (2023)')
ax = fig.add_subplot(gs[0, 3]); U(ax, 'CCGG', 'G', 'DDDD', '(d) Hybrid conv. + global attn.', 'convolution local, attention at coarse levels')
# (b) UNETR: transformer encoder at single resolution
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis('off')
ax.text(0.1, 10.9, '(b) Transformer encoder', fontsize=7.8, fontweight='bold', va='top', color=INK)
ax.text(0.1, 10.1, 'UNETR (2022); Swin UNETR (2022)', fontsize=6.2, va='top', color=GRY)
for k in range(12):
    box(ax, 0.5, 8.6 - k * 0.6, 2.4, 0.5, fc=TC['T'], ec='white', lw=0.4, r=0.08, z=3)
ax.text(1.7, 9.35, 'patch tokens', ha='center', fontsize=6, color=INK)
for k in range(12):
    if k in (2, 5, 8, 11): ax.text(3.05, 8.85 - k * 0.6, f'z{k+1}', fontsize=5.6, va='center', color=TC['T'])
hs = [3.0, 2.3, 1.7, 1.2]; ys = [7.3, 5.5, 4.0, 2.8]
for l in range(4):
    xd = 9.4 - l * 1.05 - 0.75
    box(ax, xd, ys[l] - hs[l] / 2, 0.75, hs[l], fc=DEC, ec='white', lw=0.4, r=0.1, z=3)
    src = 8.85 - [2, 5, 8, 11][l] * 0.6
    arrow(ax, 3.7, src, xd, ys[l], c=GRY, lw=0.6, ms=4, ls=(0, (2, 2)))
    if l < 3:
        arrow(ax, xd - 1.05 + 0.55, ys[l + 1] + hs[l + 1] / 2, xd + 0.37, ys[l] - hs[l] / 2, c=DEC, lw=0.7, ms=4)
ax.text(6.3, 1.1, 'reshaped and up-sampled\nto each decoder level', ha='center', fontsize=6, color=GRY)

# legend row
lax = fig.add_axes([0.02, 0.475, 0.96, 0.03]); lax.axis('off'); lax.set_xlim(0, 100); lax.set_ylim(0, 1)
for i, (t, lab) in enumerate([('C', 'conv. encoder block'), ('D', 'conv. decoder block'), ('T', 'Transformer layer'),
                              ('W', 'shifted-window attention'), ('G', 'global attention in conv. stage')]):
    x = 1 + i * 20
    box(lax, x, 0.1, 1.4, 0.8, fc=TC[t], r=0.1); lax.text(x + 2.0, 0.5, lab, va='center', fontsize=6.4)

# (e) memory
ax = fig.add_subplot(gs[1, :])
n = np.array([32, 48, 64, 96, 128, 160, 192, 256])
N = (n / 4) ** 3
glob = N ** 2 * 4 / 1e9
win = N * 7 ** 3 * 4 / 1e9
glob8 = (n / 8) ** 6 * 4 / 1e9
ax.semilogy(n, glob, 'o-', color=NEW, lw=1.2, ms=3.5, label='global attention, $4^3$-sample tokens')
ax.semilogy(n, glob8, 's--', color='#7B4FA0', lw=1.2, ms=3.5, label='global attention at 1/8 resolution (coarse stage)')
ax.semilogy(n, win, '^-', color=DEC, lw=1.2, ms=3.5, label='$7^3$-token windows, $4^3$-sample tokens')
for g, lab in ((24, '24 GB'), (80, '80 GB')):
    ax.axhline(g, color=GRY, lw=0.6, ls=':'); ax.text(264, g, lab + ' accelerator', va='center', fontsize=6.2, color=GRY, bbox=dict(fc='white', ec='none', pad=0.5))
ax.set_xlabel('cubic patch size $n$ (samples per axis)'); ax.set_ylabel('memory of one attention matrix (GB)')
ax.set_xticks(n); ax.set_xlim(28, 315)
ax.legend(frameon=False, loc='upper left', fontsize=6.5)
ax.text(-0.05, 1.03, '(e)', transform=ax.transAxes, fontsize=9, fontweight='bold')
save(fig, 'Fig07_attention_placement')
print(dict(zip(n, glob.round(3))))
