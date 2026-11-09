import csv, re, json
P='/home/claude/src/zip/reveiwpaper/harvest_candidates.csv'
rows=list(csv.DictReader(open(P,encoding='utf-8-sig')))
UNET=re.compile(r'\bu[\s\-]?net|unet|v-net\b|\bvnet\b|u-shaped (network|neural|architecture|encoder|cnn|convolution)|unetr|transunet|swin-?unet|nnu-?net',re.I)
ENCDEC=re.compile(r'encoder[\s\-–]+decoder',re.I)
EQ=re.compile(r'earthquake|phase pick|seismogram|teleseism|aftershock|microseism|magnitude|seismic event detection|ground motion|structural (health|response)|building|bridge|liquefaction|landslide|seismic risk|seismic vulnerab|seismic performance',re.I)
SEIS=re.compile(r'seismic|reflection|migration|post-?stack|pre-?stack|shot gather|common[- ]midpoint|impedance|full[- ]waveform|fault|horizon|salt|facies|first[- ]break|first[- ]arrival|deblend|ground[- ]roll|multiples|velocity model|reservoir',re.I)
def txt(r): return (r['title']+' '+r['abstract'])
for r in rows:
    t=txt(r)
    r['unet']=bool(UNET.search(t)); r['encdec']=bool(ENCDEC.search(t))
    r['eq']=bool(EQ.search(t)); r['seis']=bool(SEIS.search(t))
