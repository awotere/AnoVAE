
from GRU_VAE import BuildVAE,BuildData
import numpy as np
import GetSensorData as GS
import Global as G

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#エラーレート算出
def GetErrorRate(test,pred):
    sum = 0
    for i in range(G.NUM_TIMESTEPS):
        sum = sum + abs(test[0][i][0] - pred[0][i][0])

    return sum


#VAE環境を作成
vae = BuildVAE()
#ウェイトの読み込み
vae.load_weights("./data/gru_vae.h5")

def pause_plot():
    fig,(ax1,ax2) = plt.subplots(2,1)

    x = range(G.NUM_TIMESTEPS)

    true = [0]*G.NUM_TIMESTEPS
    pred = [0]*G.NUM_TIMESTEPS

    ax1.set_xlabel("time")
    ax1.set_xlim(0,G.NUM_TIMESTEPS)
    ax1.set_ylim(0,1)
    
    ax2.set_ylabel("Error Rate")
    ax2.set_xlabel("time")
    ax2.set_xlim(0,G.NUM_TIMESTEPS)
    ax2.set_ylim(0,7)

    line1, = ax1.plot(x,true,label = "original")
    line2, = ax1.plot(x,pred,label = "reconstructed")
    line3, = ax2.plot(x,G.ERROR_LOG, label="ErrorRate")
    
    ax1.legend()
    ax2.legend()
    
    while True:
        #データを作成
        X_test = GS.GetTestDataSet()
        X_pred = vae.predict(X_test)
    
        error = GetErrorRate(X_test,X_pred)
        G.ERROR_LOG = np.append(G.ERROR_LOG,error)
        G.ERROR_LOG = np.delete(G.ERROR_LOG,0)

        true = GS.TestData2GraphArray(X_test)
        pred = GS.TestData2GraphArray(X_pred)

        line1.set_data(x, true)
        line2.set_data(x, pred)
        line3.set_data(x, G.ERROR_LOG)

        plt.pause(.01)

if __name__ == "__main__":
    pause_plot()







