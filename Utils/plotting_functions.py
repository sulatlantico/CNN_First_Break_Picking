import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_single_trace_fig(cmp,trace,ams_min=0,ams_max=1001,figsize=(20,4)):
    """
    Função para plotar um único traço
    """
    fig,axes = plt.subplots(figsize=figsize)
    x = np.arange(0,len(cmp[trace]))
    axes.plot(x,cmp[trace],color='black')
    axes.fill_between(x=x,y1=cmp[trace],where=(cmp[trace]>0.0),color='black')
    axes.set_xlim(ams_min,ams_max)
    axes.set_title('Trace %i'%trace)
    return fig

def wigle_plot(data,trace_ini,trace_end,fat=5,step_xticks=5,figsize=(14,15),title=None):
    """Função para plotar gráfico wiggle
    
    Input
    
    data - traços sísmicos (arr)
    trace_ini - primeiro traço a ser plotado (int)
    trace_end - último traço a ser plotado (int)
    fat - fator de escala para ajustar o plot (float ou int)
    step_xticks - intervalo a ser mostrado no xticks
    figsize - tamanho da figura (tuple(int,int))
    """
    
    fig,axes = plt.subplots(figsize=figsize)
    axes.set_title(title,fontsize=16)
    axes.invert_yaxis()
    
    for i in range(trace_ini,trace_end):
        x = (data[i]/data[i].std())+fat*i
        y = np.arange(len(data[i]))
        
        axes.plot(x,y,color='black')
        axes.fill_betweenx(y=y,x1=x,x2=fat*i,where=(x>fat*i),color='black')
        
    axes.set_xticks(np.arange(trace_ini,trace_end+step_xticks,step_xticks)*fat)
    labels = [str(i) for i in np.arange(trace_ini,trace_end+step_xticks,step_xticks)]
    axes.set_xticklabels(labels)
    
    axes.set_xlabel('Número do traço')
    axes.set_ylabel('Número da amostra')
    
    return fig,axes

def plot_tiro(pred_dataset,predictions, number_tiro,save=False, color='green'):
    """
    Função para plotar um traço com a predição de primeira quebra
    """
    tiros = [i for i in pred_dataset.loc[:,'N_tiro'] ]
    df = pd.DataFrame(data = predictions)
    df.insert(pred_dataset.shape[1]-1,'N_tiro', tiros) #1001
    number_tiro = number_tiro
    caos = df[df.N_tiro==number_tiro]
    data = caos.loc[:,caos.columns != 'N_tiro']
    time = np.arange(0.004, 1024*0.004+0.004, 0.004)
    extent = [caos.index[0], caos.index[-1], int(time[-1]), int(time[0])]
    data_arr = np.asarray(data)
    shot = pred_dataset.loc[pred_dataset[pred_dataset.N_tiro==number_tiro].index]
    shot_drop = shot.drop(columns=['N_tiro'])
    arg_max = np.asarray([np.argmax(data_arr[i]) for i in range(data_arr.shape[0])])
    arg_max = arg_max*0.004
    #plotting
    plt.figure(dpi=300)
    plt.imshow(shot_drop.T,cmap='gray_r',aspect='auto',vmin=-0.000001,vmax=0.000001,
               extent=extent) 
#     plt.plot(np.arange(0,len(df.loc[df[df.N_tiro==number_tiro].index])),arg_max, color=color)
    plt.plot(np.arange(caos.index[0], caos.index[-1]+1),arg_max,color=color, scaley=False)
    plt.xlabel(f'Trace Number: {number_tiro}')
    plt.ylabel('Time (s)')
    if save == True:
        plt.savefig(f'U-Net_Tiro_{number_tiro}.png')
    else:
        pass
    plt.show()

# def plot_tiro(pred_dataset,predictions, number_tiro,save=False, color='green'):
#     """
#     Função para plotar um traço com a predição de primeira quebra
#     """
#     tiros = []
#     for i in pred_dataset.loc[:,'N_tiro']:
#         tiros.append(i)
#     df = pd.DataFrame(data = predictions)
#     df.insert(pred_dataset.shape[1],'N_tiro', tiros) #1001 #pred_dataset.shape[1]-1
#     number_tiro = number_tiro
#     caos = df[df.N_tiro==number_tiro]
#     data = caos.loc[:,caos.columns != 'N_tiro']
#     data_arr = np.asarray(data)
#     arg_max = []
#     for i in range(data_arr.shape[0]):
#         max_arg = np.argmax(data_arr[i])
#         arg_max.append(max_arg)
#     #plotting
#     plt.figure(dpi=300)
#     plt.imshow(pred_dataset.loc[pred_dataset[pred_dataset.N_tiro==number_tiro].index].T,cmap='gray_r',
#                aspect='auto',vmin=-0.000001,vmax=0.000001, 
#                extent=[caos.index[0],caos.index[-1],time[0],time[-1]]) 
#     plt.plot(np.arange(0,len(df.loc[df[df.N_tiro==number_tiro].index])),arg_max, color=color)
#     plt.ylim(800,0)
#     if save == True:
#         plt.savefig(f'U-Net_Tiro_{number_tiro}.png')
#     else:
#         pass
#     plt.show()
    
def plot_wigle_tiro(pred_dataset,predictions, number_tiro,fat=15,
                    save=False,filename='predict', step_xticks=5,figsize=(14,15),
                    title=None, color='green'):
    
    tiros = [i for i in pred_dataset.loc[:,'N_tiro'] ]
    df = pd.DataFrame(data = predictions)
    df.insert(pred_dataset.shape[1]-1,'N_tiro', tiros)#1001
    number_tiro = number_tiro
    caos = df[df.N_tiro==number_tiro]
    data = caos.loc[:,caos.columns != 'N_tiro']
    data_arr = np.asarray(data)
    arg_max = np.asarray([np.argmax(data_arr[i]) for i in range(data_arr.shape[0])])
    wigle_plot(pred_dataset[pred_dataset.N_tiro==number_tiro][np.arange(0,pred_dataset.shape[1]-1)].values,0, #0,1001
              pred_dataset[pred_dataset.N_tiro==number_tiro][np.arange(0,pred_dataset.shape[1]-1)].shape[0], fat=fat, #0,1001
              step_xticks=step_xticks,figsize=figsize,title=f'Tiro_{number_tiro}')
    plt.plot(np.arange(0,len(df.loc[df[df.N_tiro==number_tiro].index]))*fat,arg_max, color=color)
    plt.xlim(0,100*15)
    plt.ylim(400,0)
    if save == True:
        plt.savefig(f'U-Net_Tiro_{number_tiro}.png')
    else:
        pass
    plt.show()

