import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# ---------------------------------------------------------------- Fig. 3 task pairs
seis, imp, fmask = section(nz=160, nx=160, seed=11, throw=10)
rng = np.random.default_rng(1)
noise = gaussian_filter(rng.normal(0, 1, seis.shape), 0.6)
noisy = seis + 0.45 * noise / noise.std() * seis.std() * 2.2
keep = rng.random(seis.shape[1]) > 0.45
keep[70:88] = False
dec = seis * keep[None, :]
vmod, rec, tt, tfb = fd_data()
gather = gained(rec, tt)
fbm = fb_mask(tfb, tt)
fm = gaussian_filter(fmask, 0.7); fm = fm / fm.max()

fig = plt.figure(figsize=(7.1, 4.5))
gs = fig.add_gridspec(2, 6, wspace=0.08, hspace=0.42, left=0.02, right=0.98, top=0.9, bottom=0.04)
kw = dict(cmap='gray_r', vmin=-1, vmax=1, aspect='auto', interpolation='bilinear')
pairs = [
    ('(a) Random-noise attenuation', 'R', (noisy, kw), (seis, kw)),
    ('(b) Trace interpolation', 'R', (dec, kw), (seis, kw)),
    ('(c) Impedance inversion', 'M', (seis, kw), (np.log(imp), dict(cmap='viridis', aspect='auto', vmin=np.percentile(np.log(imp),1), vmax=np.percentile(np.log(imp),99)))),
    ('(d) Gather to velocity model', 'M', (gather, dict(cmap='gray_r', vmin=-0.22, vmax=0.22, aspect='auto')), (vmod, dict(cmap='turbo', aspect='auto', vmin=1500, vmax=4600))),
    ('(e) First-break picking', 'S', (gather, dict(cmap='gray_r', vmin=-0.22, vmax=0.22, aspect='auto')), (fbm, dict(cmap='Purples', vmin=-0.3, vmax=1.3, aspect='auto', interpolation='nearest'))),
    ('(f) Fault segmentation', 'S', (seis, kw), (fm, dict(cmap='Purples', vmin=0, vmax=1, aspect='auto'))),
]
roles = {'R': 'Restoration', 'M': 'Mapping', 'S': 'Segmentation'}
for i, (title, role, inp, tgt) in enumerate(pairs):
    r, c = divmod(i, 3)
    a1 = fig.add_subplot(gs[r, 2 * c]); a2 = fig.add_subplot(gs[r, 2 * c + 1])
    a1.imshow(inp[0], **inp[1]); a2.imshow(tgt[0], **tgt[1])
    if i == 4:
        xs = np.arange(0, len(tfb), 8)
        a1.plot(xs, tfb[xs] / (tt[1] - tt[0]), 'o', color=NEW, ms=1.3, mew=0)
    for a, lab in ((a1, 'input'), (a2, 'target')):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.5); s.set_color('#9CA3AF')
        a.text(0.5, -0.02, lab, transform=a.transAxes, ha='center', va='top', fontsize=6.5, color=GRY)
    a1.text(0.0, 1.13, title, transform=a1.transAxes, fontsize=8, fontweight='bold', ha='left', va='bottom', color=INK)
    a1.text(0.0, 1.03, roles[role], transform=a1.transAxes, fontsize=7, ha='left', va='bottom', color=ROLE[role])
    # arrow between
    fig.patches.append(FancyArrowPatch((0, 0), (0, 0)))
save(fig, 'Figure_3')

# ---------------------------------------------------------------- Fig. 5 skip connections
seis, imp, fmask = section(nz=256, nx=256, seed=4, throw=12, f=28)
f = 16
low = zoom(zoom(gaussian_filter(seis, 0.0), 1 / f, order=1), f, order=1)[:256, :256]
# proper: emulate 4 x (2x maxpool-free decimation with antialias) then upsample
down = seis.copy()
for _ in range(3):
    down = gaussian_filter(down, 1.0)[::2, ::2]
up = zoom(down, 8, order=3)[:256, :256]
resid = seis - up

fig = plt.figure(figsize=(7.1, 3.9))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], height_ratios=[1, 1], wspace=0.32, hspace=0.55,
                      left=0.03, right=0.99, top=0.93, bottom=0.1)
ims = [(seis, '(a) Input section'), (up, '(b) Down/up-sampled (1/8)'), (resid, '(c) Detail carried by skips')]
for i, (im, t) in enumerate(ims):
    ax = fig.add_subplot(gs[:, i])
    ax.imshow(im, cmap='gray_r', vmin=-0.8, vmax=0.8, aspect='auto', interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.5); s.set_color('#9CA3AF')
    ax.set_title(t, loc='left', fontsize=8, fontweight='bold', pad=3)
    if i == 0:
        ax.set_ylabel('time samples'); ax.set_xlabel('traces')
# spectra
ax = fig.add_subplot(gs[0, 3])
def spec(a):
    S = gaussian_filter(np.abs(np.fft.rfft(a, axis=0)).mean(1), 1.5); return S / S.max()
k = np.fft.rfftfreq(256)
ax.plot(k, spec(seis), color=ENC, lw=1.1, label='input')
ax.plot(k, spec(resid), color=NEW, lw=0.9, ls=':', label='skip residual')
ax.plot(k, spec(up), color=BOT, lw=1.1, label='down/up-sampled')
ax.axvline(0.5 / 8, color=NEW, lw=0.7, ls='--')
ax.text(0.135, 0.62, 'Nyquist after\n3 down-samplings', color=NEW, fontsize=6.5, va='top')
ax.set_xlabel('vertical wavenumber (cycles/sample)'); ax.set_ylabel('amplitude')
ax.set_xlim(0, 0.25); ax.legend(frameon=False, loc='upper right', fontsize=6.5)
ax.set_title('(d) Vertical amplitude spectra', loc='left', fontsize=8, fontweight='bold', pad=3)
# receptive field
ax = fig.add_subplot(gs[1, 3])
def rf(dil=False):
    r, s, out = 1, 1, [1]
    for lvl in range(5):
        for c in range(2):
            d = 2 ** c if (dil and lvl >= 3) else 1
            r += (3 - 1) * d * s
        out.append(r)
        if lvl < 4:
            r += (2 - 1) * s; s *= 2
    return out
lv = np.arange(6)
ax.plot(lv, rf(False), 'o-', color=ENC, ms=3, lw=1.1, label='standard encoder')
ax.plot(lv, rf(True), 's--', color=NEW, ms=3, lw=1.1, label='dilated blocks at levels 4–5')
ax.set_xticks(lv); ax.set_xticklabels(['in', '1', '2', '3', '4', '5'])
ax.set_xlabel('encoder level'); ax.set_ylabel('receptive field (samples)')
ax.legend(frameon=False, fontsize=6.5, loc='upper left')
ax.set_title('(e) Theoretical receptive field', loc='left', fontsize=8, fontweight='bold', pad=3)
save(fig, 'Figure_5')
print(rf(False), rf(True))
