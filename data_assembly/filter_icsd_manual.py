import pandas as pd
import numpy as np
import re
from timeit import default_timer
from pymatgen.ext.matproj import MPRester

m = MPRester('N1mVV71oB4tEQmdioxJE')

def get_element(x):
    return re.split(r'(\d*\.?\d+)', x)[:-1:2]

def get_species(data):
    species = set()
    for entry in data.itertuples():
        species = species.union(entry.elements)
    return sorted(list(species))

infile = '../data/icsd_manual_process.csv'
outfile = '../data/icsd_manual_query.csv'
start_idx = 52000

if start_idx:
    icsd = pd.read_csv(outfile)
    icsd['elements'] = icsd['elements'].apply(eval)
    icsd['mp_id'] = icsd['mp_id'].apply(eval)
else:
    icsd = pd.read_csv(infile)
    icsd['formula'] = icsd['formula'].apply(str)
    icsd['icsd'] = icsd['icsd'].apply(int)
    icsd['elements'] = icsd['formula'].map(lambda x: get_element(x))
    icsd['mp_id'] = np.empty((len(icsd), 0)).tolist()

ti = default_timer()    # checkpoint time
checkpoint = 1000
L = len(icsd.iloc[start_idx:])

for i, entry in enumerate(icsd.iloc[start_idx:].itertuples()):
    query = m.query(criteria={"elements": {"$all": entry.elements}, "nelements": len(entry.elements)}, 
                    properties=["task_id", "icsd_ids", "pretty_formula"])
    
    # record query
    if len(query):
        l = []
        for k in query:
            if icsd.at[entry.Index,'icsd'] in k['icsd_ids']:
                l += [k['task_id']]
        
        icsd.at[entry.Index, 'mp_id'] += l
        
    
    # save checkpoint
    if (i+1)%checkpoint == 0:
        icsd.to_csv(outfile, index=False)
        tf = default_timer()
        print(i+1, '/', L, ':', tf-ti, 'sec.')
        ti = tf
        
# final save
icsd.to_csv(outfile, index=False)
