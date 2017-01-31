import pandas as pd

file = 'allResults.h5'
typ = ['Euler','Double','SweptGPU']

st = pd.HDFStore(file)
brick = []

for s in sorted(st.keys()):
    inter = st[s].xs(typ[0]).xs(typ[1]).xs(typ[2]).min(axis=1)
    brick.append(pd.DataFrame(inter))

df = pd.concat(brick, axis=1)
df.columns = sorted(st.keys())
df.to_csv('EulerDoublePast.csv')