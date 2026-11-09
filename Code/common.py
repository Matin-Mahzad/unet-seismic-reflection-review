"""Shared style, palette and synthetic-seismic utilities for all review figures."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

OUT = '/home/claude/work/ms/figs/'

mpl.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'mathtext.fontset': 'stixsans', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# palette (colour-blind-safe, consistent across figures)
ENC = '#3B6FB6'      # encoder / processing
DEC = '#2A9D8F'      # decoder / imaging-inversion
BOT = '#E9A93B'      # bottleneck
NEW = '#C2185B'      # what changed / attention
GRY = '#6B7280'      # neutral lines
LGT = '#EEF1F5'      # light panel fill
INK = '#1F2933'      # text
ROLE = {'R': '#3B6FB6', 'M': '#E07B39', 'S': '#7B4FA0'}   # restoration, mapping, segmentation
DOM = {'Processing': '#3B6FB6', 'Imaging & inversion': '#E07B39', 'Interpretation': '#7B4FA0'}


def save(fig, name):
    fig.savefig(OUT + name + '.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(OUT + name + '.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def panel_label(ax, s, x=-0.02, y=1.02, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right', **kw)


def box(ax, x, y, w, h, fc, ec=None, lw=0.8, r=0.02, text=None, fs=7, tc='white', z=2, **kw):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc,
                       ec=ec or fc, lw=lw, zorder=z, **kw)
    ax.add_patch(p)
    if text is not None:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=z + 1)
    return p


def arrow(ax, x0, y0, x1, y1, c=GRY, lw=0.9, ms=7, ls='-', z=1, cs='arc3,rad=0'):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw,
                        linestyle=ls, zorder=z, connectionstyle=cs, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


# ----------------------------------------------------------------- synthetic seismic
def ricker(f, dt, n=None):
    n = n or int(2.0 / (f * dt)) | 1
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def layered_impedance(nz, nx, rng, nlayers=40, smooth=0.0):
    """Random layered impedance (1D log replicated laterally)."""
    z = np.sort(rng.choice(np.arange(5, nz - 5), nlayers, replace=False))
    imp = np.full(nz, 4.0e6)
    val = 4.0e6
    for zi in z:
        val = val * np.exp(rng.normal(0.02, 0.12))
        imp[zi:] = val
    imp = imp + np.linspace(0, 3e6, nz)
    return np.tile(imp[:, None], (1, nx))


def fold_and_fault(field, rng, amp=6.0, fault=True, throw=9.0, dip_x=(0.42, 0.62), order=1, return_mask=False):
    """Apply gentle folding and one normal fault to a 2D (z, x) field."""
    nz, nx = field.shape
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing='ij')
    shift = amp * np.sin(2 * np.pi * xx / nx * rng.uniform(0.8, 1.4) + rng.uniform(0, np.pi)) * (0.4 + 0.6 * zz / nz)
    src_z = zz - shift
    mask = np.zeros_like(field)
    if fault:
        x0, x1 = dip_x[0] * nx, dip_x[1] * nx
        xf = x0 + (x1 - x0) * zz / nz          # fault trace x(z)
        hang = xx > xf
        src_z = np.where(hang, src_z - throw, src_z)
        d = np.abs(xx - xf)
        mask = (d < 1.0).astype(float)
    out = map_coordinates(field, [np.clip(src_z, 0, nz - 1), xx], order=order, mode='nearest')
    if return_mask:
        return out, mask
    return out


def section(nz=192, nx=192, f=30.0, dt=0.002, seed=3, fault=True, throw=9.0, amp=6.0):
    """Synthetic section in the spirit of FaultSeg3D: deform a random reflectivity model, then convolve."""
    rng = np.random.default_rng(seed)
    nz0, pad = nz, 40
    nz = nz + 2 * pad
    nl = nz
    # normal-incidence reflectivity r_i (realistic magnitude) and the impedance it implies,
    # ln Z_{i+1} - ln Z_i = ln[(1 + r_i)/(1 - r_i)]  (exact inverse of r = (Z2 - Z1)/(Z2 + Z1))
    r1 = rng.normal(0, 1, nl) * (rng.random(nl) < 0.35) * np.exp(rng.normal(0, 0.5, nl))
    r1 = np.clip(0.04 * r1, -0.3, 0.3)
    refl = np.tile(r1[:, None], (1, nx))
    lnz = np.cumsum(np.log((1 + r1) / (1 - r1)))
    imp0 = np.tile(np.exp(lnz + np.log(4.0e6))[:, None], (1, nx))
    st = np.random.default_rng(seed + 100)
    refl_d, mask = fold_and_fault(refl, np.random.default_rng(seed + 1), amp=amp, fault=fault, throw=throw, order=1, return_mask=True)
    imp = fold_and_fault(imp0, np.random.default_rng(seed + 1), amp=amp, fault=fault, throw=throw, order=1)
    w = ricker(f, dt)
    seis = np.apply_along_axis(lambda c: np.convolve(c, w, mode='same'), 0, refl_d)
    sl = slice(pad, pad + nz0)
    seis, imp, mask = seis[sl], imp[sl], mask[sl]
    seis /= np.percentile(np.abs(seis), 99.5)
    return np.clip(seis, -1.2, 1.2), imp, mask


def shot_gather(nt=300, nx=96, dt=0.004, dx=25.0, seed=5, f=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * dt
    x = (np.arange(nx) - nx // 2) * dx
    d = np.zeros((nt, nx))
    w = ricker(f, dt)
    # direct / refracted first arrival (linear), reflections (hyperbolic)
    t_fb = 0.04 + np.abs(x) / 1800.0
    vel = 1700.0
    t0s = np.sort(rng.uniform(0.15, nt * dt * 0.9, 9))
    for i, t0 in enumerate(t0s):
        v = vel + 350 * i
        tt = np.sqrt(t0 ** 2 + (x / v) ** 2)
        a = rng.uniform(0.4, 1.0) * rng.choice([-1, 1])
        for j in range(nx):
            k = int(round(tt[j] / dt))
            if k < nt and tt[j] > t_fb[j]:
                d[k, j] += a
    for j in range(nx):
        k = int(round(t_fb[j] / dt))
        if k < nt:
            d[k, j] += 1.4
    d = np.apply_along_axis(lambda c: np.convolve(c, w, mode='same'), 0, d)
    d *= (1 + 3 * (t[:, None] / t[-1]) ** 2)       # gain
    d /= np.abs(d).max()
    fbmask = (t[:, None] >= t_fb[None, :]).astype(float)
    return d, fbmask, t_fb


def velocity_model(nz=128, nx=128, seed=7):
    rng = np.random.default_rng(seed)
    v = np.zeros((nz, nx))
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing='ij')
    bounds = np.sort(rng.uniform(10, nz - 10, 6))
    val = 1600.0
    v[:] = val
    for b in bounds:
        val += rng.uniform(250, 500)
        bnd = b + 6 * np.sin(2 * np.pi * xx / nx * 1.3 + 0.5)
        v = np.where(zz > bnd, val, v)
    # salt-like body
    cx, cz = nx * 0.55, nz * 0.62
    body = ((xx - cx) / (nx * 0.16)) ** 2 + ((zz - cz) / (nz * 0.22)) ** 2 < 1
    body &= zz > nz * 0.42
    v = np.where(body, 4400.0, v)
    return v


def fd_data():
    """Velocity model, finite-difference shot gather shifted so that t = 0 at the source-wavelet peak,
    and eikonal first-arrival times (see physics.py)."""
    import os
    D = np.load(os.path.join(os.path.dirname(__file__), 'fd_cache.npz'))
    v, rec, t, tt, delay = D['v'], D['rec'], D['t'], D['tt'], float(D['delay'])
    k = int(round(delay / (t[1] - t[0])))
    rec = rec[k:]; t = t[: rec.shape[0]]
    return v, rec, t, tt


def gained(rec, t):
    g = rec * (t[:, None] + 0.05)          # first-order spherical-divergence style gain
    return g / np.percentile(np.abs(g), 99.0)


def fb_mask(tt, t):
    return (t[:, None] >= tt[None, :]).astype(float)
