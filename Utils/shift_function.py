import numpy as np
import pandas as pd

def shift(dataset):
    traces_shift = []
    n_tiro_list = []
    n_picks_list = []
    for i in range(len(dataset)*3):
        idx = np.random.choice(np.arange(0,4889))
        trace = dataset[np.arange(0,1001)].iloc[idx]
        n_pick = dataset.iloc[idx]['n_picks']
        n_tiro = dataset.iloc[idx]['N_tiro']
        shift = np.random.choice(np.arange(10,301))
        trace_roll = np.roll(trace, shift)
        trace_roll[:shift] = 0
        n_pick += shift
        traces_shift.append(trace_roll)
        n_tiro_list.append(n_tiro)
        n_picks_list.append(n_pick)
    df = pd.DataFrame(data=traces_shift)
    df.insert(1001,'N_tiro',n_tiro_list)
    df.insert(1002,'n_picks',n_picks_list)
    dataset_shifted = pd.merge(dataset,df,how='outer')
    return dataset_shifted