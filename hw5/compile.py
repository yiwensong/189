import pandas as pd
import numpy as np

import glob

PATH = 'out/multispam/'

files = glob.iglob(PATH + '*')

n = 0.
df = None
for f in files:
    if df is None:
        df = pd.DataFrame.from_csv(f)
    else:
        df += pd.DataFrame.from_csv(f)
    n += 1.

df = np.round(df/n)
df['category'] = np.array(df['category'],dtype='int')
df.to_csv('out/multispam/spam.csv')

print df
