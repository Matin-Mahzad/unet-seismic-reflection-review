import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from collections import defaultdict

GEN = {0: ('G0 Plain U-Net (baseline)', '#4B5563'), 1: ('G1 Volumetric', ENC), 2: ('G2 Better blocks', DEC),
       3: ('G3 Channel and spatial attention', '#E07B39'), 4: ('G4 Redesigned skip pathways', '#9A7B1C'),
       5: ('G5 Transformer and state-space', '#7B4FA0'), 6: ('G6 U-Net as a component', NEW)}
generic = [(2015.0, 'FCN', 0), (2015.35, 'U-Net', 0), (2015.75, 'SegNet', 0), (2016.2, '3D U-Net', 1), (2016.55, 'V-Net', 1),
           (2016.0, 'ResNet', 2), (2017.2, 'DenseNet', 2), (2017.6, 'Transformer', 5), (2017.3, 'pix2pix', 6),
           (2018.05, 'SE block', 3), (2018.3, 'Attention U-Net', 3), (2018.55, 'UNet++', 4), (2018.8, 'ResUNet', 2), (2018.15, 'ASPP', 2),
           (2018.45, 'Deep image prior', 6), (2019.3, 'Noise2Void', 6), (2020.2, 'UNet 3+', 4), (2020.4, 'MultiResUNet', 2),
           (2020.75, 'DDPM', 6), (2021.05, 'ViT', 5), (2021.3, 'TransUNet', 5), (2021.55, 'Swin', 5), (2021.8, 'nnU-Net', 6),
           (2022.1, 'UNETR', 5), (2022.35, 'Swin UNETR', 5), (2022.6, 'MAE', 6), (2023.0, 'Swin-Unet', 5), (2023.4, 'SAM', 6),
           (2023.8, 'Mamba', 5), (2024.2, 'U-Mamba', 5)]
seismic = [(2018.6, 'Patch CNNs: faults, salt', 0), (2019.1, 'First breaks; interpolation', 0), (2019.4, 'Velocity-model building', 0),
           (2019.25, 'FaultSeg3D (3D)', 1), (2019.6, 'SaltSeg (3D)', 1), (2020.3, 'Residual U-Net faults', 2), (2020.6, '3D UNet++ faults', 4),
           (2021.1, 'Residual U-Net denoising', 2), (2021.4, 'Blind-spot denoising', 6), (2021.7, 'Deep-prior 3D enhancement', 6),
           (2022.0, '2.5D attention U-Net faults', 3), (2022.25, 'Nested residual U-Net', 4), (2022.5, 'MultiResUNet deblending', 2),
           (2022.75, 'Reparametrised FWI', 6), (2023.1, 'Transformer dual U-Net', 5), (2023.35, '2.5D Transformer U-Net', 5),
           (2023.6, 'Diffusion interpolation', 6), (2024.1, 'Swin-UNETR faults', 5), (2024.4, 'Diffusion denoising, ground roll', 6),
           (2024.7, 'Lightweight 3D U-Nets', 2), (2025.1, 'Seismic foundation model', 6), (2025.4, 'TransUNet horizons, karst', 5),
           (2025.75, 'Promptable geobody models', 6), (2026.2, 'Mamba U-Net horizons', 5)]

fig, ax = plt.subplots(figsize=(7.1, 5.6))
ax.set_xlim(2014.6, 2027.6); ax.axis('off')
ax.axhline(0, color=INK, lw=1.0, zorder=1)
for y in range(2015, 2027):
    ax.plot([y, y], [-0.35, 0.35], color=INK, lw=0.8)
    ax.text(y, -0.9, str(y), ha='center', va='top', fontsize=6.8, color=INK)

ext = [0]
def place(items, sign):
    ext[0] = 0
    items = sorted(items)
    lanes = []  # last x end per lane
    for x, lab, g in items:
        wlab = 0.09 * len(lab) + 0.3
        for li in range(30):
            if li >= len(lanes): lanes.append(-1e9)
            if lanes[li] < x - 0.02:
                lanes[li] = x + wlab; break
        y = sign * ((1.4 if sign > 0 else 2.3) + 1.0 * li)
        ext[0] = max(ext[0], abs(y))
        c = GEN[g][1]
        ax.plot([x, x], [0, y - sign * 0.35], color=c, lw=0.45, alpha=0.55, zorder=1)
        ax.plot(x, 0, 'o', ms=3.2, color=c, zorder=3)
        ax.text(x, y, lab, ha='left', va='center', fontsize=5.9, color=c,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=c, lw=0.5), zorder=4)
place(generic, +1); top = ext[0]
place(seismic, -1); bot = ext[0]
ax.set_ylim(-bot - 1.6, top + 1.6)
ax.text(2014.65, top + 1.2, 'Generic architectures (computer vision, medical imaging)', fontsize=7.5, fontweight='bold', color=INK)
ax.text(2014.65, -bot - 1.2, 'Representative seismic adoptions (Sections 6–8)', fontsize=7.5, fontweight='bold', color=INK)
handles = [plt.Line2D([], [], marker='o', ls='', color=c, ms=4, label=l) for l, c in GEN.values()]
ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=6.3, ncol=1, bbox_to_anchor=(1.0, 0.97))
save(fig, 'Figure_6')
