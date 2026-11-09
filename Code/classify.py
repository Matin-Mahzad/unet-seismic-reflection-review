"""Keyword classification of screened records into tasks and architectural families.

Run `python classify.py` with the harvest path set in load.py to rebuild the corpus;
the figure scripts import only the label definitions below.
"""
import re
from collections import Counter, defaultdict
NONREF=re.compile(r'phase pick|earthquake|microseism|ground[- ]penetrating|\bGPR\b|landslide|InSAR|seismogram|teleseism|DAS\b|distributed acoustic|acoustic emission|building|concrete|ultrasound|medical|tumou?r|retinal|crack|MRI\b|\bCT\b|volcan|seismolog|steel|shake table|fragility|subway|timber|masonry|beam|column|wall|damper|bearing|dispersion curve|topographic effect',re.I)
KEEP_TYPES={'article','conference-paper','conference-abstract','preprint','dissertation','review','book-chapter','report'}
TASKS=[ # (key,label,group,regex)
('denoise','Random-noise attenuation','Processing',r'denois|random noise|noise (attenuation|suppression|removal|reduction)|signal[- ]to[- ]noise'),
('coherent','Coherent noise, multiples & deblending','Processing',r'ground[- ]roll|surface[- ]wave (attenuat|suppress|remov)|multiple(s)? (attenuat|suppress|remov|elimin|predict)|demultiple|deblend|blended|interference noise|swell noise|coherent noise|surface[- ]related multiple|linear noise|erratic noise|footprint|ghost|multiple subtraction|adaptive subtraction|demultiple'),
('interp','Interpolation & reconstruction','Processing',r'interpolat|reconstruct|missing trace|regulari[sz]ation of seismic|data recovery|undersampl|irregular(ly)? sampl|trace (recovery|completion)'),
('firstbreak','First-break picking','Processing',r'first[- ]break|first[- ]arrival'),
('velpick','Velocity analysis & picking','Processing',r'velocity (pick|analys|spectr)|semblance|stacking velocity|nmo'),
('resol','Resolution, bandwidth & low frequencies','Processing',r'resolution enhanc|super[- ]resolution|bandwidth|low[- ]frequenc(y|ies) (extrap|reconstr|predict|recover|data)|deconvolution|spectral (enhanc|broaden|extrap)|frequency (extension|extrapolation)|resolution (improv|increas)'),
('migration','Migration & imaging','Imaging & inversion',r'migrat|inverse hessian|demigrat|\brtm\b|least[- ]squares (reverse|migration)|image[- ]domain|imaging gather|common[- ]image'),
('vmb','Velocity-model building & FWI','Imaging & inversion',r'velocity model|full[- ]waveform|\bfwi\b|velocity inversion|tomograph|inversionnet'),
('impedance','Impedance & reservoir properties','Imaging & inversion',r'impedance|petrophysic|porosity|lithology|reservoir (propert|characteri|parameter)|elastic (parameter|propert)|avo\b|pre-?stack inversion|well[- ]log'),
('fault','Faults','Interpretation',r'fault'),
('horizon','Horizons & geologic time','Interpretation',r'horizon|geologic(al)? time|rgt\b|stratigraph(ic)? (surface|interpret)|unconformit'),
('salt','Salt bodies','Interpretation',r'\bsalt'),
('facies','Facies & stratigraphy','Interpretation',r'facies|seismic stratigraph|sequence stratigraph|lithofacies'),
('geobody','Channels, karst & other geobodies','Interpretation',r'channel(s)? (detect|identif|segment|interpret|characteri|extract|recogni)|paleochannel|karst|cave|geobod|gas chimney|mass[- ]transport|igneous|volcanic|reef|carbonate build|fluvial'),
('dhi','Hydrocarbon indicators & 4D/CO2','Interpretation',r'bright spot|direct hydrocarbon|gas (detect|identif|zone|reservoir)|hydrocarbon (detect|indicat|predict)|time[- ]lapse|4d|co2|plume'),
]
FAM=[ # key,label,regex  (ordered by evolution)
('plain','Plain 2D U-Net',None),
('3d','3D / volumetric',r'\b3[- ]?d\b|three[- ]dimensional|volumetric|v-?net'),
('res','Residual / dense blocks',r'residual|resunet|res-?u-?net|dense(ly)?[- ]?(connect|block|u-?net|net)|r2u|inception'),
('multi','Multi-scale / dilated / ASPP',r'multi[- ]?scale|dilat|atrous|aspp|pyramid'),
('nested','Nested / full-scale skips (UNet++, 3+)',r'unet\+\+|u-net\+\+|unet 3\+|unet3\+|nested|full[- ]scale|deep supervision'),
('att','Attention (gates, SE, CBAM)',r'attention|squeeze[- ]and[- ]excit|\bse[- ]block|cbam|scse'),
('trans','Transformer / Swin',r'transformer|swin|self[- ]attention|vision transformer|\bvit\b|unetr|transunet'),
('gan','Adversarial (GAN)',r'adversarial|\bgan\b|pix2pix|cyclegan|generative adversarial'),
('diff','Diffusion / score-based',r'diffusion model|denoising diffusion|ddpm|ddim|score[- ]based|diffusion probabilistic|diffusion-based'),
('ssl','Self-/semi-/unsupervised',r'self[- ]supervis|semi[- ]supervis|unsupervis|blind[- ]spot|noise2|zero[- ]shot|deep image prior|contrastive|pretext|masked auto'),
('phys','Physics-guided / -informed',r'physics[- ](guided|informed|constrained|driven|based)|wave[- ]equation|forward model(l)?ing (operator|constraint)|closed[- ]loop|cycle[- ]consist'),
('unc','Uncertainty (Bayesian, MC-dropout)',r'uncertaint|bayesian|monte carlo dropout|mc[- ]dropout|ensemble'),
('found','Foundation / large pretrained',r'foundation model|segment anything|\bsam\b|large[- ]scale pretrain|pre-?trained (on|with) (large|massive)'),
('light','Lightweight / efficient',r'lightweight|light-weight|depthwise|mobilenet|pruning|knowledge distillation|parameter[- ]efficient|low[- ]parameter'),
('mamba','State-space (Mamba)',r'mamba|state[- ]space'),
]
TR=[(k,l,g,re.compile(rx,re.I)) for k,l,g,rx in TASKS]
FR=[(k,l,re.compile(rx,re.I) if rx else None) for k,l,rx in FAM]
TLAB={k:l for k,l,g,_ in TASKS}; TGRP={k:g for k,l,g,_ in TASKS}; FLAB={k:l for k,l,_ in FAM}


def classify(rows):
    corpus=[]; seen=set()
    for r in rows:
        if not r['unet'] or r['status'] not in ('likely','maybe','maybe-no-abstract'): continue
        if r['type'] not in KEEP_TYPES: continue
        t=r['title']+' '+r['abstract']
        if NONREF.search(r['title']) or (NONREF.search(t) and not re.search(r'reflection|post-?stack|pre-?stack|fault|horizon|salt|facies|impedance|migration|shot gather|full[- ]waveform',t,re.I)): continue
        if not r['seis']: continue
        key=re.sub(r'\W+','',r['title'].lower())[:80]
        if key in seen: continue
        seen.add(key)
        try: y=int(r['year'])
        except: continue
        if y<2015 or y>2026: continue
        r['y']=y
        r['tasks']=[k for k,l,g,rx in TR if rx.search(t)]
        fams=[k for k,l,rx in FR if rx and rx.search(t)]
        r['fams']=fams
        corpus.append(r)
    return corpus


if __name__ == '__main__':
    from load import load_rows
    corpus = classify(load_rows())
    print(len(corpus), 'records')
