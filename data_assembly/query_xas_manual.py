import pandas as pd
import numpy as np
from timeit import default_timer
from pymatgen.ext.matproj import MPRester

m = MPRester('N1mVV71oB4tEQmdioxJE')

def format_mpid(x):
    if x.startswith('['):
        return eval(x)
    else:
        return x

infile = '../data/icsd_manual_query.csv'
outfile = '../data/xas_manual_query.csv'
start_idx = 0

if start_idx:
    df = pd.read_csv(outfile)
    df['elements'] = df['elements'].apply(eval)
    df['mp_id'] = df['mp_id'].apply(format_mpid)
else:
    df = pd.read_csv(infile)
    df['elements'] = df['elements'].apply(eval)
    df['mp_id'] = df['mp_id'].apply(eval)
    df = df[df['mp_id'].str.len() > 0].reset_index(drop=True)
    df['structure'] = np.empty((len(df), 0)).tolist()
    df['spectra'] = np.empty((len(df), 0)).tolist()


ti = default_timer()    # checkpoint time
checkpoint = 1000
L = len(df.iloc[start_idx:])

for i, entry in enumerate(df.iloc[start_idx:].itertuples()):
    for mp_id in entry.mp_id:
        l = dict.fromkeys(entry.elements)
        for absorbing_atom in entry.elements:
            try: xas = m.get_xas_data(mp_id, absorbing_atom)
            except:
                # element is missing xas spectrum
                break
            else:
                struct = xas['spectrum']['structure']
                l[absorbing_atom] = {'x': xas['spectrum']['x'], 'y': xas['spectrum']['y']}

        if not any(v == None for v in l.values()):
            df.at[entry.Index, 'structure'] = struct
            df.at[entry.Index, 'spectra'] = l
            df.at[entry.Index, 'mp_id'] = mp_id
            break


    # save checkpoint
    if (i+1)%checkpoint == 0:
        df.to_csv(outfile, index=False)
        tf = default_timer()
        print(i+1, '/', L, ':', tf-ti, 'sec.')
        ti = tf

# final save
df.to_csv(outfile, index=False)
