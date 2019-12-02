
import math
import os
import numpy as np
from keras.layers import Input, InputLayer, Dense, RepeatVector, Lambda, TimeDistributed
# from keras.layers import GRU
from keras.layers import CuDNNGRU as GRU  # GPU用
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import adam
from keras import backend as K

from sklearn.preprocessing import MinMaxScaler
# データの準備

import Global as G

def BuildData(dir,min_val,max_val):

    #データ読み込み(全サンプル数)の配列
    X = np.loadtxt(dir,encoding = "utf-8-sig")

    X_average = np.average(X)
    #最小値を0にして0-1に圧縮

    clamp = lambda x,min_val_a,max_val_a: min(max_val_a, max(x, min_val_a))
    X = np.array(list(map(lambda x:clamp((x-min_val)/(max_val-min_val),0,1),X)))
    
    #一次元配列から二次元行列に変換(None, 1)
    X = X.reshape(-1,1)

    #全サンプル数定義
    sample_size = X.shape[0] - G.TIMESTEPS

    #(サンプル数,timestep)の行列
    Xr = np.zeros((sample_size, G.TIMESTEPS))
    Xr2 = np.zeros((sample_size, G.TIMESTEPS))

    #timestep分スライスして格納
    for i in range(sample_size):
        Xr[i] = X[i:i + G.TIMESTEPS].T

        if i == 0:
            Xr2[0] = (np.array([X_average]) + X[: G.TIMESTEPS]).T
            continue

        Xr2[i] = X[i + 1 : i + 1 + G.TIMESTEPS].T




    #kerasに渡す形(sample,timestep,features)に変換
    Xr = np.expand_dims(Xr, axis=2)
    Xr2 = np.expand_dims(Xr2, axis=2)

    #内部処理用のデータセット

    return Xr,Xr2




'''
非推奨
def BuildVAE():
    """
    入力(input)
    ↓
    GRU(encoder)
    ↓
    内部状態
    ↓   ↓
    mean, log_var
    ↓
    zをサンプリング(ここまでencoder)
    ↓（このzを復元された内部状態だとして）
    GRU(decoder)
    ↓
    全結合層(出力)
    戻り値
     model
    """
    #LATENT_DIM = G.LATENT_DIM

    def sampling(args):
        """
        z_mean, z_log_var=argsからzをサンプリングする関数
        戻り値
            z (tf.tensor):サンプリングされた潜在変数
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        # K.exp(0.5 * z_log_var)が分散に標準偏差になっている
        # いきなり標準偏差を求めてしまっても構わないが、負を許容してしまうのでこのようなトリックを用いている
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # encoderの定義
    encoder_inputs = Input(shape=(G.TIMESTEPS, 1))
    _, h = GRU(G.Z_DIM, return_state=True)(encoder_inputs)
    z_mean = Dense(G.Z_DIM, name='z_mean')(h)  # z_meanを出力
    z_log_var = Dense(G.Z_DIM, name='z_log_var')(h)  # z_sigmaを出力

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(G.Z_DIM,), name='z')([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    print("encoderの構成")
    encoder.summary()
    # encoder部分は入力を受けて平均、分散、そこからランダムサンプリングしたものの3つを返す

    # decoderの定義

    #1
    #latent_inputs = RepeatVector(G.NUM_TIMESTEPS)(z)
    #x = GRU(LATENT_DIM, return_sequences=True)(latent_inputs)
    #outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    #2
    #latent_inputs = Input(shape=(G.LATENT_DIM,), name='z_sampling')
    #x = Dense(G.INTERMIDIATE_DIM, activation='relu')(latent_inputs)
    #outputs = Dense(G.NUM_TIMESTEPS, activation='sigmoid')(x)
    
    #decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()

    #techblog.exawizards.com/entry/2018/11/09/145402

    #3
    from keras.layers import concatenate


    decoder_inputs = Input(shape=(G.TIMESTEPS,1), name='z_sampling')
    # (N,TIMESTEPS,1)

    overlay_x = RepeatVector(G.TIMESTEPS)(z)
    # (N,TIMESTEPS,LATENT_DIM)

    #入力は[データ,z]
    actual_input_x = concatenate([decoder_inputs,overlay_x],2)

    #zから初期状態hを決定
    initial_h = Dense(G.Z_DIM, activation="tanh")(z)

    zd = GRU(G.Z_DIM,return_sequences=True)(actual_input_x, initial_state=initial_h)


    outputs = Dense(G.NUM_TIMESTEPS, activation='sigmoid')(zd)

    #x = Dense(G.INTERMIDIATE_DIM,activation = "sigmoid")(z)
    #outputs = Dense(G.NUM_TIMESTEPS,activation = "sigmoid")(x)

    #decoder = Model(z, outputs, name='decoder')
    decoder = Model(decoder_inputs, outputs, name='decoder')
    print("decoderの構成")
    decoder.summary()
    
    #まとめ
    vae = Model([encoder_inputs,decoder_inputs], outputs, name='VAE')

    # 損失関数をこのモデルに加える
    def loss(inputs, outputs):
        """
        損失関数の定義
        """
        from keras.losses import binary_crossentropy
        z_mean, z_log_var, _ = encoder(inputs)
        reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= 1 * G.TIMESTEPS
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        lam = 0.01 #そのままじゃうまく行かなかったので重み付け
        return K.mean((1-lam)*reconstruction_loss + lam*kl_loss)

    vae.add_loss(loss(encoder_inputs, outputs))
    print("vaeの構成")
    vae.summary()

    return vae

'''
