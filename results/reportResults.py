import numpy as np
import glob
import pandas as pd
fps = glob.glob('results/*/*/g*')


for fp in fps:

    data = pd.DataFrame(np.load(fp))

    fp = fp.split("/")

    cat = fp[2]
    route = fp[3][-5]

    print("{}_{}: {} | Episodes: {}/150000".format(cat,route,float(data.rolling(1000,1000).mean().max()),data.shape[0]))
