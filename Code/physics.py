"""Physics-based synthetic data for the review figures.

velocity_model_salt : layered sediments uplifted and dragged against a salt diapir.
fd_shot_gather      : 2D constant-density acoustic finite-difference modelling
                      (2nd-order time, 8th-order space, absorbing sponge boundaries).
first_break_mask    : first-arrival times picked on the noise-free modelled data.
conv_image          : convolutional-model depth image of the same model
                      (normal-incidence reflectivity from impedance, Gardner density).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

def ricker(f, dt, n=None):
    n = n or int(2.0 / (f * dt)) | 1
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)

def velocity_model_salt(nz=150, nx=240, dx=8.0, seed=7):
    rng = np.random.default_rng(seed)
    zz, xx = np.meshgrid(np.arange(nz) * dx, np.arange(nx) * dx, indexing='ij')
    X = nx * dx; Zm = nz * dx
    # salt top: Gaussian dome rising from the base
    xc, w, top = 0.56 * X, 0.13 * X, 0.42 * Zm
    salt_top = Zm - (Zm - top) * np.exp(-np.abs((xx - xc) / w) ** 2.6)
    salt = zz > salt_top
    # sediment interfaces: gentle dip + drag/uplift towards the diapir
    base = np.sort(rng.uniform(0.08, 0.95, 7)) * Zm
    uplift = 0.22 * Zm * np.exp(-((xx - xc) / (1.9 * w)) ** 2)
    v = np.full((nz, nx), 1800.0)
    vel = 1800.0
    for i, b in enumerate(base):
        vel += rng.uniform(220, 420) * (1 if i % 3 != 1 else -0.5)
        bnd = b + 0.03 * (xx - X / 2) - uplift * (b / Zm)
        v = np.where(zz > bnd, vel, v)
    v = np.where(salt, 4500.0, v)
    return v

def fd_shot_gather(v, dx=8.0, dt=6e-4, T=1.6, f=15.0, sx=None, sz=3, pad=120):
    """Constant-density acoustic wave equation p_tt = v^2 lap(p) + s(t) delta(x - xs).

    Second-order time stepping, eighth-order spatial stencil, no periodic wrap-around:
    the grid is surrounded by a 120-cell Cerjan taper and a zero halo. Receivers along
    the line z = sz. Returns the pressure record and its time axis (time zero = source start).
    """
    nz, nx = v.shape
    vp = np.pad(v, pad, mode='edge')
    NZ, NX = vp.shape
    dist = np.zeros((NZ, NX))
    zz, xx = np.meshgrid(np.arange(NZ), np.arange(NX), indexing='ij')
    dist = np.maximum.reduce([pad - zz, zz - (NZ - 1 - pad), pad - xx, xx - (NX - 1 - pad), np.zeros_like(zz)]).astype(float)
    damp = np.exp(-(0.0035 * dist) ** 2)
    c = [-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]
    h = 4
    P0 = np.zeros((NZ + 2 * h, NX + 2 * h)); P1 = P0.copy()
    k2 = (vp * dt / dx) ** 2
    nt = int(round(T / dt))
    sx = nx // 2 if sx is None else sx
    src = ricker(f, dt, int(0.2 / dt) | 1)
    rec = np.zeros((nt, nx))
    I = slice(h, h + NZ); J = slice(h, h + NX)
    for it in range(nt):
        p1 = P1[I, J]
        lap = 2 * c[0] * p1
        for k in range(1, 5):
            lap = lap + c[k] * (P1[h - k:h - k + NZ, J] + P1[h + k:h + k + NZ, J] + P1[I, h - k:h - k + NX] + P1[I, h + k:h + k + NX])
        p2 = 2 * p1 - P0[I, J] + k2 * lap
        if it < len(src):
            p2[pad + sz, pad + sx] += k2[pad + sz, pad + sx] * src[it]
        p2 *= damp
        P0[I, J] = p1 * damp
        P1[I, J] = p2
        rec[it] = p2[pad + sz, pad:pad + nx]
    t = np.arange(nt) * dt
    return rec, t


def eikonal_first_arrivals(v, dx=8.0, sx=None, sz=3):
    """First-arrival traveltimes (direct, refracted/head and diving waves) by fast marching."""
    import skfmm
    nz, nx = v.shape
    sx = nx // 2 if sx is None else sx
    phi = np.ones((nz, nx)); phi[sz, sx] = -1
    tt = skfmm.travel_time(phi, v, dx=dx, order=2)
    return np.asarray(tt)[sz, :]


def first_break_mask(rec, t, frac=0.02):
    env = np.abs(rec)
    nt, nx = rec.shape
    fb = np.zeros(nx)
    for j in range(nx):
        thr = frac * env[:, j].max()
        idx = np.argmax(env[:, j] > thr)
        fb[j] = t[idx]
    fb = gaussian_filter(fb, 1.0)
    mask = (t[:, None] >= fb[None, :]).astype(float)
    return mask, fb

def conv_image(v, f_samples=10):
    rho = 310.0 * v ** 0.25                       # Gardner
    z = v * rho
    r = np.zeros_like(z)
    r[1:] = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    w = ricker(1.0 / f_samples, 1.0, 4 * f_samples + 1)
    img = np.apply_along_axis(lambda col: np.convolve(col, w, mode='same'), 0, r)
    return img / np.abs(img).max()
