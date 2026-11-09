"""Load the OpenAlex harvest CSV and flag U-Net, encoder-decoder, earthquake and seismic-context records.

The raw harvest is not part of this repository. Set HARVEST to its path, or set the HARVEST_CSV
environment variable, before running classify.py.
"""
import csv, os, re, json
HARVEST = os.environ.get('HARVEST_CSV', 'harvest_candidates.csv')
UNET=re.compile(r'\bu[\s\-]?net|unet|v-net\b|\bvnet\b|u-shaped (network|neural|architecture|encoder|cnn|convolution)|unetr|transunet|swin-?unet|nnu-?net',re.I)
ENCDEC=re.compile(r'encoder[\s\-–]+decoder',re.I)
EQ=re.compile(r'earthquake|phase pick|seismogram|teleseism|aftershock|microseism|magnitude|seismic event detection|ground motion|structural (health|response)|building|bridge|liquefaction|landslide|seismic risk|seismic vulnerab|seismic performance',re.I)
SEIS=re.compile(r'seismic|reflection|migration|post-?stack|pre-?stack|shot gather|common[- ]midpoint|impedance|full[- ]waveform|fault|horizon|salt|facies|first[- ]break|first[- ]arrival|deblend|ground[- ]roll|multiples|velocity model|reservoir',re.I)
def txt(r): return (r['title']+' '+r['abstract'])
def load_rows(path=HARVEST):
    rows=list(csv.DictReader(open(path,encoding='utf-8-sig')))
    for r in rows:
        t=txt(r)
        r['unet']=bool(UNET.search(t)); r['encdec']=bool(ENCDEC.search(t))
        r['eq']=bool(EQ.search(t)); r['seis']=bool(SEIS.search(t))
    return rows
