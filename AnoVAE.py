# VAE関係
# ダイアログ関係
import os
import tkinter
import tkinter.filedialog
import tkinter.messagebox as MSGBOX
import numpy as np

import pyautogui

import GRU_VAE as GV
import Global as G  # 非推奨

root = tkinter.Tk()
root.geometry("0x0")
root.overrideredirect(1)
root.withdraw()

def BuildData(dir, min_val, max_val):
    # データ読み込み(全サンプル数)の配列
    X = np.loadtxt(dir, encoding="utf-8-sig")

    # 最小値を0にして0-1に圧縮
    clamp = lambda x, min_val_a, max_val_a: min(max_val_a, max(x, min_val_a))
    X = np.array(list(map(lambda x: clamp((x - min_val) / (max_val - min_val), 0, 1), X)))

    # 一次元配列から二次元行列に変換(None, 1)
    X = np.reshape(X,newshape=(-1))

    # 全サンプル数(入力csvのデータ数)
    sample_size = X.shape[0]

    # (サンプル数,timestep)の行列
    Xr = np.zeros(shape=(sample_size, G.TIMESTEPS))

    # timestep分スライスして格納
    for i in range(sample_size):
        if i < G.TIMESTEPS - 1: #(0~)
            #Xr[0]: [0, 0, 0, ... , 0   , X[0]] shape=(TIMESTEP)
            #Xr[1]: [0, 0, 0, ... , X[0], X[1]] shape=(TIMESTEP)
            #こんな配列
            zero_array = np.zeros(shape=(G.TIMESTEPS-i-1))
            Xr[i] = np.hstack((zero_array,X[:i+1]))
        else:
            Xr[i] = X[i - G.TIMESTEPS + 1:i+1]

    # kerasに渡す形(sample,timestep,features)に変換
    Xr = np.expand_dims(Xr, axis=2)

    # 内部処理用のデータセット(初期値のこと)がX
    np.reshape(X,newshape=(-1,1))

    return Xr, X


def GetFilePathFromDialog(file_types):
    # ファイル選択ダイアログの表示
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes=file_types, initialdir=iDir)

    return file

''' 非推奨
def TestData2List(x):

    sample_size = x.shape[0]
    timesteps = x.shape[1]
    #データは一次元なので(sample_size,timesteps,1) -> (sample_size,timesteps)とする
    np.reshape(x,newshape=(sample_size,timesteps))

    # x[0]  : [0    , 0     , ... , 0    , X[0]]
    # x[1]  : [0    , 0     , ... , X[0] , X[1]]
    #  :    :    :       :    ＼     :       :
    # x[-1] : [X[-t],X[-t+1], ... , X[-2], X[-1]]

    return x[:,-1]
'''

def ShowGlaph(t, r, e,dm):
    import matplotlib.pyplot as plt
    plt.subplot(3, 1, 1)
    plt.ylabel("Value")
    #plt.xlabel("time")
    plt.ylim(0, 1)

    plt.plot(range(len(t)), t, label="original")
    plt.plot(range(len(r)), r, label="reconstructed")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.ylabel("Error Rate")
    #plt.xlabel("time")
    # plt.ylim(0,10)

    plt.plot(range(len(e)), e, label="Manhattan distance")
    plt.legend()


    plt.subplot(3, 1, 3)
    plt.ylabel("Mahalanobis Distance")
    plt.xlabel("time")
    # plt.ylim(0,10)

    plt.plot(range(len(dm)), dm, label="Mahalanobis Distance")
    plt.legend()

    plt.show()


def Show_t_SNE(X):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    result = TSNE(n_components=2).fit_transform(X).T
    plt.scatter(result[0],result[1])
    plt.title("t-SNE")
    plt.show()
    return

#########################    AnoVAE    ############################

class AnoVAE:
    # メンバ変数
    vae = None
    encoder = None
    decoder = None

    load_weight_flag = False
    load_muSIGMA_flag = False

    MIN_OF_12bit = 0
    MAX_OF_12bit = 4095

    true_mu = None
    true_SIGMA = None

    # コンストラクタ
    def __init__(self):
        self.vae, self.encoder, self.decoder = self.BuildVAE()
        return

    def BuildVAE(self):
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

        # LATENT_DIM = G.LATENT_DIM

        from keras.layers import Input, Dense, RepeatVector, Lambda, TimeDistributed,concatenate
        # from keras.layers import GRU
        from keras.layers import CuDNNGRU as GRU  # GPU用
        from keras.models import Model
        from keras import backend as K

        # encoderの定義
        # (None, TIMESTEPS, 1) <- TIMESTEPS分の波形データ
        encoder_inputs = Input(shape=(G.TIMESTEPS, 1),name="encoder_inputs")

        # (None, Z_DIM) <- h
        _, h_forw = GRU(G.Z_DIM, return_state=True,name="encoder_GRU_forward")(encoder_inputs)
        _, h_back = GRU(G.Z_DIM, return_state=True,go_backwards=True, name="encoder_GRU_backward")(encoder_inputs)

        h = concatenate([h_forw,h_back],axis=1)

        # (None, Z_DIM) <- μ
        z_mean = Dense(G.Z_DIM, name='z_mean')(h)  # z_meanを出力

        # (None, Z_DIM) <- σ^ (σ = exp(log(σ^/2)))
        z_log_var = Dense(G.Z_DIM, name='z_log_var')(h)  # z_sigmaを出力

        #z導出
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

        # (None,Z_DIM)
        z = Lambda(sampling, output_shape=(G.Z_DIM,), name='z')([z_mean, z_log_var])

        #エンコーダー (1次元のTIMESTEPS分のデータ) -> ([μ,σ^,z])
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        print("encoderの構成")
        encoder.summary()

        # encoder部分は入力を受けて平均、分散、そこからランダムサンプリングしたものの3つを返す

        # decoderの定義

        from keras.layers import concatenate

        # (None, 1) <- 予測する波形の初期値
        decoder_inputs = Input(shape=(1,), name='decoder_inputs')
        input_z = Input(shape=(G.Z_DIM,),name="input_z")

        # (None, 1 + Z_DIM)
        actual_input_x = concatenate([decoder_inputs,input_z],axis=1)

        # (None, TIMESTEPS, 1 + Z_DIM) <- from z
        repeat_x = RepeatVector(G.TIMESTEPS)(actual_input_x)

        # zから初期状態hを決定
        # (None, Z_DIM)
        initial_h = Dense(G.Z_DIM, activation="tanh",name="initial_state_layer")(input_z)

        # (None, TIMESTEPS, Z_DIM)
        zd = GRU(G.Z_DIM, return_sequences=True, name="decoder_GRU")(repeat_x, initial_state=initial_h)

        # (None, TIMESTEPS, 1)
        outputs = TimeDistributed(Dense(1, activation='sigmoid'),name="output_layer")(zd)

        decoder = Model([decoder_inputs,input_z], outputs, name='decoder')
        print("decoderの構成")
        decoder.summary()

        # まとめ
        outputs = decoder([decoder_inputs,encoder(encoder_inputs)[2]])
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

            lam = G.Loss_Lambda  # そのままじゃうまく行かなかったので重み付け
            return K.mean((1 - lam) * reconstruction_loss + lam * kl_loss)

        vae.add_loss(loss(encoder_inputs, outputs))
        print("vaeの構成")
        vae.summary()

        return vae, encoder, decoder


    # メンバ関数
    def Train(self, path=None):

        # 学習データcsvファイルのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "学習データを選んでください")
            path = GetFilePathFromDialog([("csv", "*.csv"), ("すべてのファイル", "*")])

        # 学習データを作成
        X_train,X_train2 = BuildData(path, self.MIN_OF_12bit, self.MAX_OF_12bit)
        print("Trainデータ読み込み完了")

        # コンパイル
        self.vae.compile(optimizer="adam")

        # 学習
        from keras.callbacks import TensorBoard, EarlyStopping
        history = self.vae.fit([X_train,X_train2],
                     epochs=100,
                     batch_size=G.BATCH_SIZE,
                     shuffle=True,
                     validation_split=0.1,
                     callbacks=[TensorBoard(log_dir="./train_log/"), EarlyStopping(patience=10)])

        import matplotlib.pyplot as plt
        # 損失の履歴をプロット
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='lower right')
        plt.show()

        print("学習終了!")

        # W保存
        name = pyautogui.prompt(text="weight保存名を指定してください", title="AnoVAE", default="ts{0}_zd{1}_b{2}_lam{3}".format(G.TIMESTEPS, G.Z_DIM, G.BATCH_SIZE,G.Loss_Lambda))

        weight_path = "./data/weight/{0}.h5".format(name)
        self.vae.save_weights(filepath=weight_path)
        print("weightを保存しました:\n{0}", weight_path)

        #mu,sigmaを求め、保存
        # zの取得
        from keras.models import Model
        import numpy as np

        print("μと∑を取得します")

        encoder_layer = self.vae.get_layer("encoder")
        encoder = Model(encoder_layer.get_input_at(0), encoder_layer.get_output_at(0))
        mu_sigma = encoder.predict(X_train)

        #μ  (4800,25)->(25)
        mu = mu_sigma[0]
        mu = mu[G.TIMESTEPS:]
        mu = np.average(mu,axis=0)

        #σ  (4800,25)->(25)
        sigma = np.exp(mu_sigma[1]/2)
        sigma = sigma[G.TIMESTEPS:]
        sigma = np.average(sigma,axis=0)

        #∑=diag(σ)  (25,25)
        SIGMA = np.diag(sigma)

        path = "./data/mu_SIGMA/{0}.npz".format(name)
        np.savez(path,mu=mu,SIGMA=SIGMA)
        print("μΣを保存しました:\n{0}".format(path))

        print("Train終了")
        self.true_mu = mu
        self.true_SIGMA = SIGMA
        self.load_weight_flag = True
        self.load_muSIGMA_flag = True
        return

    def LoadWeight(self, path=None):

        # Wのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "weightデータを選んでください")
            path = GetFilePathFromDialog([("weight", "*.h5"), ("すべてのファイル", "*")])

        # Wの読み込み
        self.vae.load_weights(path)
        print("weightを読み込みました:\n{0}".format(path))

        self.load_weight_flag = True
        return

    def TestCSV(self, path=None):
        # Wの読み込み
        if not self.load_weight_flag:
            self.LoadWeight()
        if not self.load_muSIGMA_flag:
            self.LoadMuSIGMA()

        # テスト用csvファイルのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "testデータを選んでください")
            path = GetFilePathFromDialog([("テスト用csv", "*.csv"), ("すべてのファイル", "*")])

        # テストデータセット作成
        X_true,X_true2 = BuildData(path, self.MIN_OF_12bit, self.MAX_OF_12bit);

        print("テストデータを読み込みました:\n{0}".format(path))

        # predict

        X_reco = np.empty(shape=(0,G.TIMESTEPS))

        # z取得
        mu_list, _, z_list = self.encoder.predict(X_true)

        # X_reco取得
        count = 0
        num_space = 0
        max_len = len(X_true2)
        for x,z in zip(X_true2,z_list):

            if count > max_len/10:
                num_space += 1
                print("再構成しています... progress [{0}{1}]\r".format("■"*num_space,"＿"*(10-num_space)))
                count = 0

            # zは[1,25]
            z = np.reshape(z, (1, -1))
            x = np.reshape(x, (1, -1))
            x_reco = self.decoder.predict([x,z])
            x_reco = np.reshape(x_reco,newshape=(1,G.TIMESTEPS))
            X_reco = np.append(X_reco,x_reco,axis=0)
            count += 1


        print("再構成完了しました")

        reco_view = np.array([])
        # 表示用のX_reco -> reco
        for x_reco in X_reco[G.TIMESTEPS - 1::G.TIMESTEPS]:
            x_reco = np.reshape(x_reco,newshape=G.TIMESTEPS)

            reco_view = np.hstack((reco_view, np.reshape(x_reco, newshape=(-1))))
        # リストに変換
        true_view = list(X_true2)

        offset = int(G.TIMESTEPS/2)

        # 再構成後の再構成後のマンハッタン距離
        error = [0]*offset
        for x_true,x_reco in zip(X_true[G.TIMESTEPS-1:],X_reco[G.TIMESTEPS-1:]):

            x_true = np.reshape(x_true,newshape=G.TIMESTEPS)
            x_reco = np.reshape(x_reco,newshape=G.TIMESTEPS)

            d = 0
            for t,r in zip(x_true,x_reco):
                d += abs(t - r)
            error.append(d)
        error += [0] * offset

        # z_listをt_SNEを用いて描画
        #Show_t_SNE(z_list)  # z

        #マハラノビス距離dm
        from scipy.spatial import distance

        dm = [0]*int(G.TIMESTEPS/2)
        '''
        for z in z_list[G.TIMESTEPS:]:
            dm.append(distance.mahalanobis(z,self.true_mu,np.linalg.inv(self.true_SIGMA)))
        dm += [0]*offset
        '''
        for mu in mu_list[G.TIMESTEPS:]:
            dm.append(distance.mahalanobis(mu,self.true_mu,np.linalg.inv(self.true_SIGMA)))
        dm += [0]*offset

        print("true,reco,error,dmデータ作成完了しました")

        return true_view, reco_view, error ,dm

    def LoadMuSIGMA(self):

        MSGBOX.showinfo("AnoVAE","μ∑を指定してください")
        path = GetFilePathFromDialog([("μ∑", "*.npz"), ("すべてのファイル", "*")])
        npz = np.load(path)
        self.true_mu = npz["mu"]
        self.true_SIGMA = npz["SIGMA"]
        self.load_muSIGMA_flag = True

        return



def main():
    vae = AnoVAE()
    if MSGBOX.askyesno("AnoVAE", "AnoVAEに学習させますか？"):
        vae.Train()
    else:
        vae.LoadWeight()
        vae.LoadMuSIGMA()

    # for path in glob.iglob("./data/Sin*Test*.csv"):
    #    t,r,e = vae.TestCSV(path)
    #    ShowGlaph(t,r,e)

    t, r, e, dm = vae.TestCSV()
    ShowGlaph(t, r, e,dm)
    return


if __name__ == "__main__":
    main()
