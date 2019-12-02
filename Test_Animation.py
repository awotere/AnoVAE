
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
vae.load_weights("./MNIST_gru_vae/MNIST_gru_vae.h5")

fig,(ax1,ax2) = plt.subplots(2,1)

def plot(data):
    #前のグラフを消去
    ax1.cla()
    ax2.cla()

    #データを作成
    X_test = GS.GetTestDataSet()
    X_pred = vae.predict(X_test)
    
    error = GetErrorRate(X_test,X_pred)
    G.ERROR_LOG = np.append(G.ERROR_LOG,error)
    G.ERROR_LOG = np.delete(G.ERROR_LOG,0)

    true = GS.TestData2GraphArray(X_test)
    pred = GS.TestData2GraphArray(X_pred)

    #グラフを作成     
    ax1.set_xlabel("time")
    ax1.set_xlim(0,G.NUM_TIMESTEPS)
    ax1.set_ylim(0,1)

    ax1.plot(range(G.NUM_TIMESTEPS),true,label = "original")
    ax1.plot(range(G.NUM_TIMESTEPS),pred,label = "reconstructed")
    ax1.legend()
    
    ax2.set_ylabel("Error Rate")
    ax2.set_xlabel("time")
    ax2.set_xlim(0,G.NUM_TIMESTEPS)

    ax2.plot(range(G.NUM_TIMESTEPS),G.ERROR_LOG, label="ErrorRate")
    ax2.legend()

ani = animation.FuncAnimation(fig, plot, interval=1)
plt.show()







