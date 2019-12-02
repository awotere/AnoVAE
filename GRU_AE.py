import math
import os

import numpy as np
from keras.layers import Dense, RepeatVector
# from keras.layers import GRU
from keras.layers import CuDNNGRU as GRU #GPU用
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard, EarlyStopping

def BuildData():
    DATA_DIR = "./data/"
    Ldata = np.loadtxt(DATA_DIR + "BPF7ch(learn_data).csv", encoding="utf-8-sig")
    Tdata = np.loadtxt(DATA_DIR + "createData.csv", encoding="utf-8-sig")

    print(Ldata)
    print(Tdata)

    global NUM_TIMESTEPS 
    NUM_TIMESTEPS = 500

    scaler = MinMaxScaler(feature_range = (0,1),copy = False)

    Ldata = Ldata.reshape(-1,1)    
    Tdata = Tdata.reshape(-1,1)

    Ldata = scaler.fit_transform(Ldata)
    Tdata = scaler.fit_transform(Tdata)

    sample_sizeL = Ldata.shape[0] - NUM_TIMESTEPS
    sample_sizeT = Tdata.shape[0] - NUM_TIMESTEPS

    X = np.zeros((sample_sizeL, NUM_TIMESTEPS))
    for i in range(sample_sizeL):
        X[i] = Ldata[i:i + NUM_TIMESTEPS].T

    Y = np.zeros((sample_sizeT, NUM_TIMESTEPS))
    for i in range(sample_sizeT):
        Y[i] = Tdata[i:i + NUM_TIMESTEPS].T

    # ただし、kerasに投げるときは一工夫いる、なぜなら、(samples, timesteps, features)の形になっている必要があるからだ。
    X_train = np.expand_dims(X, axis=2)
    X_test = np.expand_dims(Y, axis=2)
    #(140064, 192, 1)(サンプル数, 時間幅, 次元数)

    print("train", X_train.shape, "test", X_test.shape)
    np.save("./data/X_train.npy", X_train)
    np.save("./data/X_test.npy", X_test)

    return X_train,X_test




 #https://aotamasaki.hatenablog.com/entry/2018/09/16/175646
# AutoEncoder ネットワーク構築
def BuildAE():
    """
    入力
    ↓
    GRU(encoder)
    ↓
    内部状態 (latent)
    ↓
    GRU(decoder)
    ↓
    全結合層(出力)

    のようなシンプルなネットワーク

    戻り値
     model
    """
    LATENT_DIM = 20
    model = Sequential()
    model.add(GRU(LATENT_DIM, input_shape=(NUM_TIMESTEPS, 1)))
    model.add(RepeatVector(NUM_TIMESTEPS))
    model.add(GRU(LATENT_DIM, return_sequences=True))
    model.add(Dense(1))
    model.summary()

    return model


if __name__ == "__main__":

    print("="*20+"preparating the data..."+"="*20)
    X_train, X_test = BuildData()
    print("="*20+"summary of this model"+"="*20)

    ae = BuildAE()
    ae.compile(loss="mean_squared_error", optimizer="adam",
                metrics=["mean_squared_error"])

    ae.fit(X_train, X_train,
            epochs=30,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,
            callbacks=[TensorBoard(log_dir="./LOG"), EarlyStopping(patience=2)])
    ae.save('./LOG/GRU_AE.hdf5')

    
    # 推論
    X_pred = ae.predict(X_test)

    import matplotlib.pyplot as plt
    # データをプロット
    for true, pred in zip(X_test[::NUM_TIMESTEPS], X_pred[::NUM_TIMESTEPS]):
        plt.plot(range(true.shape[0]), true)
        plt.plot(range(pred.shape[0]), pred)
        plt.ylabel("AD")
        plt.xlabel("time")
        plt.show()



    
