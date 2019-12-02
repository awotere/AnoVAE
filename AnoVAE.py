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


def GetFilePathFromDialog(file_types):
    # ファイル選択ダイアログの表示
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes=file_types, initialdir=iDir)

    return file


def TestData2List(x):
    list = []

    sample_size = x.shape[0]
    num_timestep = x.shape[1]

    for x_pos in range(0, sample_size - num_timestep, num_timestep):
        list += [x[x_pos][y] for y in range(num_timestep)]
    else:
        mod = sample_size % num_timestep
        list += [x[x_pos][y] for y in range(mod)]

    return list


def ShowGlaph(t, r, e,dm):
    import matplotlib.pyplot as plt
    plt.subplot(3, 1, 1)
    plt.ylabel("Value")
    #plt.xlabel("time")
    plt.legend()
    plt.ylim(0, 1)

    plt.plot(range(len(t)), t, label="original")
    plt.plot(range(len(r)), r, label="reconstructed")


    plt.subplot(3, 1, 2)
    plt.ylabel("Error Rate")
    #plt.xlabel("time")
    # plt.ylim(0,10)
    plt.legend()

    plt.plot(range(len(e)), e, label="ErrorRate")


    plt.subplot(3, 1, 3)
    plt.ylabel("Mahalanobis Distance")
    plt.xlabel("time")
    # plt.ylim(0,10)
    plt.legend()

    plt.plot(range(len(dm)), dm, label="Mahalanobis Distance")

    plt.show()


def Show_t_SNE(X):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    result = TSNE(n_components=2).fit_transform(X).T
    plt.scatter(result[0],result[1])
    plt.title("t-SNE")
    plt.show()
    return

def MahalanobisDistance(mu,SIGMA,z):
    # d = √(z-μ)T∑(z-μ)
    z_mu = z - mu
    inv_SIGMA = np.linalg.inv(SIGMA)
    x = np.dot(z_mu.T,inv_SIGMA)

    return np.sqrt(np.dot(x, z_mu))

#########################    AnoVAE    ############################

class AnoVAE:
    # メンバ変数
    vae = None

    load_weight_flag = False
    load_muSIGMA_flag = False

    MIN_OF_12bit = 0
    MAX_OF_12bit = 4095

    true_mu = None
    true_SIGMA = None

    # コンストラクタ
    def __init__(self):
        self.vae = self.BuildVAE()
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

        import keras.backend as K
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

        # encoderの定義
        # (None, TIMESTEPS, 1) <- TIMESTEPS分の波形データ
        encoder_inputs = Input(shape=(G.TIMESTEPS, 1),name="encoder_inputs")

        # (None, Z_DIM) <- h
        _, h = GRU(G.Z_DIM, return_state=True, name="encoder_GRU")(encoder_inputs)

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

        # (None, TIMESTEPS, 1) <- TIMESTEPS分の波形データ
        decoder_inputs = Input(shape=(G.TIMESTEPS, 1), name='decoder_inputs')

        # (None, TIMESTEPS, Z_DIM) <- from z
        overlay_x = RepeatVector(G.TIMESTEPS)(z)

        # (None, TIMESTEPS, 1 + Z_DIM)
        actual_input_x = concatenate([decoder_inputs, overlay_x], 2)

        # zから初期状態hを決定
        # (None, Z_DIM)
        initial_h = Dense(G.Z_DIM, activation="tanh")(z)

        # (None, TIMESTEPS, Z_DIM)
        zd = GRU(G.Z_DIM, return_sequences=True)(actual_input_x, initial_state=initial_h)

        # (None, TIMESTEPS, 1)
        outputs = TimeDistributed(Dense(1, activation='sigmoid'))(zd)

        #decoder = Model(decoder_inputs, outputs, name='decoder')
        #print("decoderの構成")
        #decoder.summary()

        # まとめ
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

            lam = 0.01  # そのままじゃうまく行かなかったので重み付け
            return K.mean((1 - lam) * reconstruction_loss + lam * kl_loss)

        vae.add_loss(loss(encoder_inputs, outputs))
        print("vaeの構成")
        vae.summary()

        return vae


    # メンバ関数
    def Train(self, path=None):

        # 学習データcsvファイルのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "学習データを選んでください")
            path = GetFilePathFromDialog([("csv", "*.csv"), ("すべてのファイル", "*")])

        # 学習データを作成
        X_train,X_train2 = GV.BuildData(path, self.MIN_OF_12bit, self.MAX_OF_12bit)
        print("Trainデータ読み込み完了")

        # コンパイル
        self.vae.compile(optimizer="adam")

        # 学習
        from keras.callbacks import TensorBoard, EarlyStopping
        self.vae.fit([X_train,X_train2],
                     epochs=100,
                     batch_size=G.BATCH_SIZE,
                     shuffle=True,
                     validation_split=0.1,
                     callbacks=[TensorBoard(log_dir="./train_log/"), EarlyStopping(patience=10)])
        print("学習終了!")

        # W保存
        name = pyautogui.prompt(text="weight保存名を指定してください", title="AnoVAE", default="ts{0}_id{1}_ld{2}_b{3}".format(G.TIMESTEPS, G.INTERMIDIATE_DIM, G.Z_DIM, G.BATCH_SIZE))

        weight_path = "./data/weight/{0}.h5".format(name)
        self.vae.save(weight_path)
        print("weightを保存しました:\n{0}", weight_path)

        #mu,sigmaを求め、保存
        # zの取得
        from keras.models import Model
        import numpy as np

        print("μと∑を取得します")
        encoder = Model(self.vae.input, self.vae.get_layer("encoder").get_output_at(0))
        mu_sigma = encoder.predict(X_train)


        #μ  (4800,20)->(20)
        mu = mu_sigma[0]
        #mu = np.average(mu,axis=0)

        #σ  (4800,20)
        sigma = np.exp(mu_sigma[1]/2)

        #追加
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2)
        mu = tsne.fit_transform(mu)
        mu = np.average(mu, axis=0)

        sigma = tsne.fit_transform(sigma)

        #∑=diag(σ)  (20,20)
        SIGMA = np.cov(sigma,rowvar=False)

        np.savez("./data/mu_SIGMA/{0}".format(name),mu=mu,SIGMA=SIGMA)

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

        # テストデータと再構成データ作成
        X_true = GV.BuildData(path, self.MIN_OF_12bit, self.MAX_OF_12bit);
        print("テストデータを読み込みました:\n{0}".format(path))

        X_reco = self.vae.predict(X_true);
        print("再構成完了しました")

        # リストに変換
        true = TestData2List(X_true)
        reco = TestData2List(X_reco)

        # エラーレート計算
        error = []
        for i in range(len(true)):

            sum = 0
            for j in range(max(0, i - G.TIMESTEPS), i):
                sum += abs(true[j] - reco[j])
            error.append(sum)

        # zの取得
        from keras.models import Model

        encoder = Model(self.vae.input, self.vae.get_layer("encoder").get_output_at(0))

        z_list = (encoder.predict(X_true))[2]

        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2)
        z_list = tsne.fit_transform(z_list)

        #Show_t_SNE(z[0])  # z_mu
        #Show_t_SNE(np.exp(z[1]/2))  # z_log_var
        #Show_t_SNE(z[2])  # z

        from scipy.spatial import distance

        dm = []
        for z in z_list:
            dm.append(distance.mahalanobis(z,self.true_mu,np.linalg.inv(self.true_SIGMA)))

        #Show_t_SNE(z_list)

        print("true,reco,error,dmデータ作成完了しました")

        return true, reco, error ,dm

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
