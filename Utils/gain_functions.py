import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from seispro import agc
from torch import from_numpy, Tensor
from os.path import join


def spherical_divergence(dataset,directory,t0=2,plotting=False, pred_data=True):
    v_min,v_max= 1700,6000
    t_min,t_max, t0 = 0.004, 4.004, t0
    t = np.arange(0.004,1001*0.004,0.004)
    alpha = (v_max-v_min)/(t_max-t_min)
    v = alpha*t+v_min
    traces = dataset[dataset.columns[:1001]]
    traces_arr = np.asarray(traces)
    traces_gain = traces.values*((v**2)*t)/((v[np.where(t==t0)]**2)*t0).T
    if plotting == True:
        plot(traces_arr, traces_gain)
    else:
        pass
    if pred_data == False:
        save_data(traces_gain,dataset,directory)
    else:
        pass
    return traces_gain

def gain_agc(dataset, directory,plotting=False):
    data_gain = np.asarray(dataset[np.arange(0,1001)])
    data_gain_torch = Tensor(data_gain)
    data_gain_torch = data_gain_torch[None,:,:]

    data_agc, scaling = agc(data_gain_torch, time_window_side_len=350)
    dataset_agc = data_agc.reshape(data_agc.shape[1],data_agc.shape[2])
    if plotting == True:
        plot(data_gain,dataset_agc)
    else:
        pass
    save_data(dataset_agc,dataset,directory)
    return dataset_agc

def plot(data_without_gain,data_with_gain,n_traces=10):
    for i in range(0,n_traces):
        plt.figure(figsize=(8,5))
        plt.subplot(121)
        plt.plot(data_without_gain[i])
        plt.title('Trace Without Gain')
        plt.subplot(122)
        plt.plot(data_with_gain[i])
        plt.title('Trace with Gain')
        
def save_data(traces_arr,dataset,directory):
    #Normalizing Data
    traces_arr.std(axis=1).shape
    trace_norm =(traces_arr.T/traces_arr.std(axis=1)).T
    X,y = trace_norm, np.zeros_like(trace_norm)

    for i in range(len(dataset.n_picks)):
        y[i,int(dataset.n_picks.values[i]):]=1
    print(f'Formato de X {X.shape}')
    print(f'Formato de y {y.shape}')
    print('=================================')
    inputs = [np.asarray(X)[i] for i in range(X.shape[0])]
    labels = [np.asarray(y)[i] for i in range(X.shape[0])]

    print(f'Total de traços: {len(inputs)}')
    print(f'Total de rótulos: {len(labels)}')

    X_true, Y_true = np.asarray(inputs),np.asarray(labels)
    print('================================')
    print(f'Inputs Shape: {X_true.shape}')
    print(f'Labels Shape: {Y_true.shape}')
    X_true = np.float32(X_true)
    Y_true = np.float32(Y_true)
    np.savez_compressed(join(directory,'dataset_train.npz'), arr_0 = X_true, arr_1 = Y_true)
    print(f'Your data is now saved in:{join(directory,"dataset_train.npz")}')