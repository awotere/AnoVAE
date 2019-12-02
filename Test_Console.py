
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
    for i in range(G.NUM_TIMESTEPS):
        sum = sum + abs(test[0][i][0] - pred[0][i][0])

    return sum

count = 0
test_count = 0
import time

t_s = time.time()

while True:
        #データを作成


        X_test = GS.GetTestDataSet()
        X_pred = vae.predict(X_test)
    
        error = GetErrorRate(X_test,X_pred)

        if count >= 1000:
            
            #時間確認

            print("-"*7,"TimeTest",test_count,"-"*7)

            t = time.time()
            t_start = t
            X_test = GS.GetTestDataSet()
            print("GetTestDataSet()         : ",time.time() - t)
            
            t = time.time()
            X_pred = vae.predict(X_test)
            print("vae.predict(_)           : ",time.time() - t)
    
            t = time.time()
            error = GetErrorRate(X_test,X_pred)
            print("GetErrorRate(_,_)        : ",time.time() - t)

            print("TIME                     : ",time.time() - t_start,"/step")
            print("TIME(1000step)           : ",time.time() - t_s,"/1000step\n")
            count = 0
            test_count += 1

            t_s = time.time()

        count += 1
