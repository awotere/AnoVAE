import random
import Global as G
import numpy as np

#ランダム
def GetRndData():
    return random.uniform(G.DATA_MIN,G.DATA_MAX)

#ファイルから取得
def GetFileData():
    if GetFileData.counter < len(G.TEST_FILE) - 1:
        GetFileData.counter += 1
    return G.TEST_FILE[GetFileData.counter]
GetFileData.counter = 0

def GetSensorData():
    return 0

def CreateDataSetFromCSV(dir):
    return



#テストデータ
def GetTestDataSet():

    #センサーデータをどうにかして取得
    x = GetFileData()

    #[min,max] => [0,1]にクランプ
    x = G.Clamp((x-G.DATA_MIN)/(G.DATA_MAX-G.DATA_MIN),0,1)

    #X_LOGの先頭に最新のデータを挿入
    G.X_LOG = np.append(x,G.X_LOG)
    G.X_LOG = np.delete(G.X_LOG,len(G.X_LOG)-1)

    X = np.expand_dims(G.X_LOG,axis = 0)
    X = np.expand_dims(X,axis = 2)

    return X

#グラフ表示用に変換(3次元)
def TestData2GraphArray(x):

    r = []
    for i in range(G.NUM_TIMESTEPS):
        r.insert(0,x[0][i][0])

    return r


if __name__ == "__main__":
    import time
    import os

    while 1:

        print(GetData())
        time.sleep(0.1)
        os.system("cls")
