import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from classify import TASKS, TGRP, TLAB
import csv

# the 754 labelled records released in Data/ (classify.py rebuilds them from the raw harvest)
TKEY = {l: k for k, l, g, _ in TASKS}
corpus = []
with open(os.path.join(HERE, os.pardir, 'Data', 'corpus_754_records.csv'), encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        corpus.append({'y': int(r['year']),
                       'tasks': [TKEY[x.strip()] for x in r['tasks'].split(';') if x.strip()],
                       'fams': [x.strip() for x in r['architecture_families'].split(';') if x.strip()]})
import json
from collections import Counter

# ------------------------------------------------------------ Fig. 2
years = list(range(2018, 2027))
G = ['Processing', 'Imaging & inversion', 'Interpretation']
cnt = {g: [sum(1 for r in corpus if r['y'] == y and any(TGRP[t] == g for t in r['tasks'])) for y in years] for g in G}
tot = [sum(1 for r in corpus if r['y'] == y) for y in years]
periods = [(2018, 2020, '2018–20'), (2021, 2022, '2021–22'), (2023, 2024, '2023–24'), (2025, 2026, '2025–26')]
comp = [('att', 'Attention', NEW, '-o'), ('trans', 'Transformer / Swin', '#7B4FA0', '-s'), ('3d', '3D / volumetric', ENC, '--o'),
        ('res', 'Residual / dense', DEC, '--s'), ('ssl', 'Self-/semi-/unsupervised', BOT, '-^'), ('phys', 'Physics-guided', '#8D6E63', ':D'),
        ('diff', 'Diffusion', '#E07B39', ':v'), ('nested', 'Nested skips', GRY, '-.x')]
share = {}
for k, *_ in comp:
    share[k] = []
    for a, b, _ in periods:
        P = [r for r in corpus if a <= r['y'] <= b]
        share[k].append(100 * sum(1 for r in P if k in r['fams']) / len(P))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
x = np.arange(len(years)); w = 0.26
for i, g in enumerate(G):
    a1.bar(x + (i - 1) * w, cnt[g], w, color=DOM[g], label=g, lw=0)
a1.plot(x, tot, color=INK, lw=1, marker='o', ms=2.5, label='All records')
for xi, t in zip(x, tot):
    a1.text(xi, t + 4, str(t), ha='center', fontsize=6, color=INK)
a1.set_xticks(x); a1.set_xticklabels([str(y) if y < 2026 else '2026*' for y in years], rotation=0)
a1.set_ylabel('records'); a1.set_ylim(0, 185)
a1.legend(frameon=False, loc='upper left', fontsize=6.5)
panel_label(a1, '(a)', x=-0.08)
xp = np.arange(len(periods))
for k, lab, c, st in comp:
    a2.plot(xp, share[k], st, color=c, lw=1.1, ms=3.2, label=lab)
a2.set_xticks(xp); a2.set_xticklabels([p[2] for p in periods])
a2.set_ylabel('share of records (%)'); a2.set_ylim(0, 47); a2.set_yticks([0,10,20,30])
a2.legend(frameon=False, fontsize=6.2, ncol=2, loc='upper left', handlelength=2.2, columnspacing=0.8)
panel_label(a2, '(b)', x=-0.1)
save(fig, 'Figure_2')
print({k: [round(v, 1) for v in share[k]] for k in share})

# ------------------------------------------------------------ Fig. 10 heatmap
famorder = ['3d', 'res', 'multi', 'light', 'att', 'nested', 'trans', 'mamba', 'gan', 'diff', 'ssl', 'phys', 'unc', 'found']
famlab = {'3d': '3D / volumetric', 'res': 'Residual / dense', 'multi': 'Multi-scale / dilated', 'light': 'Lightweight',
          'att': 'Attention', 'nested': 'Nested skips', 'trans': 'Transformer', 'mamba': 'State-space', 'gan': 'Adversarial',
          'diff': 'Diffusion', 'ssl': 'Self-/semi-sup.', 'phys': 'Physics-guided', 'unc': 'Uncertainty', 'found': 'Foundation'}
taskorder = [k for k, *_ in TASKS]
M = np.array([[sum(1 for r in corpus if t in r['tasks'] and f in r['fams']) for f in famorder] for t in taskorder])
rowtot = [sum(1 for r in corpus if t in r['tasks']) for t in taskorder]
fig, ax = plt.subplots(figsize=(7.1, 4.4))
import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list('b', ['#FFFFFF', '#CFE0F3', '#3B6FB6', '#16325C'])
im = ax.imshow(np.log1p(M), cmap=cmap, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if v:
            ax.text(j, i, str(v), ha='center', va='center', fontsize=6.3, color='white' if np.log1p(v) > 2.6 else INK)
ax.set_xticks(range(len(famorder))); ax.set_xticklabels([famlab[f] for f in famorder], rotation=40, ha='right')
ax.set_yticks(range(len(taskorder))); ax.set_yticklabels([TLAB[t] for t in taskorder])
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
# group separators
for yb in (5.5, 8.5):
    ax.axhline(yb, color=INK, lw=0.8)
for xb in (3.5, 7.5, 9.5):
    ax.axvline(xb, color='white', lw=2)
for i, t in enumerate(rowtot):
    ax.text(len(famorder) - 0.35, i, str(t), ha='left', va='center', fontsize=6.5, color=INK, fontweight='bold')
ax.text(len(famorder) - 0.35, -1.0, 'total', ha='left', va='center', fontsize=6.5, color=INK, fontweight='bold')
for y, g in ((2.5, 'Processing'), (7.0, 'Imaging &\ninversion'), (11.5, 'Interpretation')):
    ax.text(-4.9, y, g, rotation=90, ha='center', va='center', fontsize=7.5, fontweight='bold', color=DOM[g.replace('\n', ' ')], transform=ax.transData, clip_on=False)
ax.set_xlim(-0.5, len(famorder) - 0.5)
save(fig, 'Figure_10')
print(M.sum(), rowtot)
