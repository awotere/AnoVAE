
from GRU_VAE import BuildVAE,BuildData
import numpy as np
import GetSensorData as GS
import Global as G


#VAE環境を作成
vae = BuildVAE()
#ウェイトの読み込み
vae.load_weights("./data/gru_vae.h5")

#エラーレート算出
def GetErrorRate(test,pred):
    sum = 0
    for i in range(G.TIMESTEPS):
        sum = sum + abs(test[0][i][0] - pred[0][i][0])

    return sum


while True:
        #データを作成
        #X_test = 
        X_pred = vae.predict(X_test)
    
        error = GetErrorRate(X_test,X_pred)


   

