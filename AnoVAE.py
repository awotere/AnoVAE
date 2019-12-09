# VAE関係
# ダイアログ関係
import os
import tkinter
import tkinter.filedialog
import tkinter.messagebox as MSGBOX
import time

import numpy as np
import pyautogui

import Global as G  # 非推奨

root = tkinter.Tk()
root.geometry("0x0")
root.overrideredirect(1)
root.withdraw()

# メモ
# (N) (N,) 要素数Nの一次元配列
# (N,M) NxM行列
# (N,M,L) NxMxL行列
# ex) (N,3,2) は3x2の行列をN個持つ配列と言える

'''
def BuildData(dir, min_val, max_val):
    # データ読み込み(サンプル数,)
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
    start_index = G.TIMESTEPS - 1
    for i in range(start_index,sample_size):
        Xr[i - start_index] = X[i - start_index:i]

    # kerasに渡す形(sample,timestep,features)に変換
    Xr = np.expand_dims(Xr, axis=2)

    # 内部処理用のデータセット(初期値のこと)がX
    np.reshape(X,newshape=(-1,1))

    return Xr, X
'''


def GetFilePathFromDialog(file_types):
    # ファイル選択ダイアログの表示
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes=file_types, initialdir=iDir)

    return file


def ShowGlaph(t, re, dkl):
    import matplotlib.pyplot as plt
    plt.subplot(3, 1, 1)
    plt.ylabel("Value")
    plt.ylim(0, 1)

    plt.plot(range(len(t)), t, label="original")
    # plt.plot(range(len(r)), r, label="reconstructed")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.ylabel("Re")

    plt.plot(range(len(re)), re, label="Manhattan")
    plt.legend()


    plt.subplot(4, 1, 3)
    plt.ylabel("Mahalanobis Distance")
    plt.xlabel("time")
    # plt.ylim(0,10)

    #plt.plot(range(len(dm)), dm, label="Mahalanobis")
    #plt.legend()

    plt.subplot(4, 1, 4)
    plt.ylabel("mu-mu Euclid Distance")
    plt.xlabel("time")
    # plt.ylim(0,10)

    plt.plot(range(len(mm)), mm, label="mu-mu Distance")
    plt.legend()

    plt.show()


def Show_t_SNE(X):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    result = TSNE(n_components=2).fit_transform(X).T
    plt.scatter(result[0], result[1])
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

    MIN_OF_12bit = 0
    MAX_OF_12bit = 4095

    # コンストラクタ
    def __init__(self):
        self.vae, self.encoder, self.decoder = self.BuildVAE()
        return

    def BuildData(self, dir, min_val, max_val):
        # データ読み込み(サンプル数,)
        X = np.loadtxt(dir, encoding="utf-8-sig")

        # 最小値を0にして0-1に圧縮
        clamp = lambda x, min_val_a, max_val_a: min(max_val_a, max(x, min_val_a))
        X = np.array(list(map(lambda x: clamp((x - min_val) / (max_val - min_val), 0, 1), X)))

        # 一次元配列から二次元行列に変換(None, 1)
        X = np.reshape(X, newshape=(-1))

        # 全サンプル数(入力csvのデータ数)
        sample_size = X.shape[0] - G.TIMESTEPS

        # (サンプル数,timestep)の行列
        Xr = np.zeros(shape=(sample_size, G.TIMESTEPS))

        # timestep分スライスして格納
        start_index = G.TIMESTEPS - 1
        for i in range(start_index, sample_size):
            Xr[i - start_index] = X[i - start_index:i + 1]

        # kerasに渡す形(sample,timestep,features)に変換
        Xr = np.expand_dims(Xr, axis=2)

        # 内部処理用のデータセット(初期値のこと)がX
        np.reshape(X, newshape=(-1, 1))

        return Xr, X[G.TIMESTEPS:]

    def BuildEncoder(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        from keras.layers import Input, Dense, Lambda, concatenate
        from keras.layers import GRU
        # from keras.layers import CuDNNGRU as GRU  # GPU用
        from keras.models import Model

        # encoderの定義
        # (None, TIMESTEPS, 1) <- TIMESTEPS分の波形データ
        encoder_inputs = Input(shape=(G.TIMESTEPS, 1), name="encoder_inputs")

        # (None, Z_DIM) <- h
        _, h_forw = GRU(G.Z_DIM, return_state=True, name="encoder_GRU_forward")(encoder_inputs)
        _, h_back = GRU(G.Z_DIM, return_state=True, go_backwards=True, name="encoder_GRU_backward")(encoder_inputs)

        h = concatenate([h_forw, h_back], axis=1)

        # (None, Z_DIM) <- μ
        z_mean = Dense(G.Z_DIM, activation="linear", name='z_mean')(h)  # z_meanを出力

        # (None, Z_DIM) <- σ^ (σ = exp(log(σ^/2)))
        z_log_var = Dense(G.Z_DIM, activation="linear", name='z_log_var')(h)  # z_sigmaを出力

        # z導出
        def sampling(args):
            from keras import backend as K
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(batch, dim))
            # K.exp(0.5 * z_log_var)が標準偏差になっている
            # いきなり標準偏差を求めてしまっても構わないが、負を許容してしまうのでこのようなトリックを用いている
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # (None,Z_DIM)
        z = Lambda(sampling, output_shape=(G.Z_DIM,), name='z')([z_mean, z_log_var])

        # エンコーダー (1次元のTIMESTEPS分のデータ) -> ([μ,σ^,z])
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        print("encoderの構成")
        encoder.summary()

        return encoder, encoder_inputs

    def BuildDecoder(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        from keras.layers import Input, Dense, RepeatVector, TimeDistributed
        from keras.layers import GRU
        # from keras.layers import CuDNNGRU as GRU  # GPU用
        from keras.models import Model

        # decoderの定義

        from keras.layers import concatenate

        # (None, 1) <- 予測する波形の初期値
        decoder_inputs = Input(shape=(1,), name='decoder_inputs')
        input_z = Input(shape=(G.Z_DIM,), name="input_z")

        # (None, 1 + Z_DIM)
        actual_input_x = concatenate([decoder_inputs, input_z], axis=1)

        # (None, TIMESTEPS, 1 + Z_DIM) <- from z
        repeat_x = RepeatVector(G.TIMESTEPS)(actual_input_x)

        # zから初期状態hを決定
        # (None, Z_DIM)
        initial_h = Dense(G.Z_DIM, activation="tanh", name="initial_state_layer")(input_z)

        # (None, TIMESTEPS, Z_DIM)
        zd = GRU(G.Z_DIM, return_sequences=True, name="decoder_GRU")(repeat_x, initial_state=initial_h)

        # (None, TIMESTEPS, 1)
        outputs = TimeDistributed(Dense(1, activation='sigmoid'), name="output_layer")(zd)

        decoder = Model([decoder_inputs, input_z], outputs, name='decoder')
        print("decoderの構成")
        decoder.summary()

        return decoder, decoder_inputs

    def BuildVAE(self):

        from keras.models import Model

        # encoder,decoderのモデルとInputレイヤーを取得
        encoder, encoder_inputs = self.BuildEncoder()
        decoder, decoder_inputs = self.BuildDecoder()

        # VAEモデル
        outputs = decoder([decoder_inputs, encoder(encoder_inputs)[2]])
        vae = Model([encoder_inputs, decoder_inputs], outputs, name='VAE')

        # 損失関数をこのモデルに加える
        def loss(inputs, outputs):
            from keras import backend as K
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

        encoder._make_predict_function()
        decoder._make_predict_function()
        # コンパイル
        vae.compile(optimizer="adam")

        return vae, encoder, decoder

    # メンバ関数
    def Train(self, path=None):

        # 学習データcsvファイルのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "学習データを選んでください")
            path = GetFilePathFromDialog([("csv", "*.csv"), ("すべてのファイル", "*")])

        # 学習データを作成
        X_train, X_train2 = self.BuildData(path, self.MIN_OF_12bit, self.MAX_OF_12bit)
        print("Trainデータ読み込み完了")

        # 学習
        t = time.time()

        from keras.callbacks import TensorBoard, EarlyStopping
        history = self.vae.fit([X_train, X_train2],
                               epochs=100,
                               batch_size=G.BATCH_SIZE,
                               shuffle=True,
                               validation_split=0.1,
                               callbacks=[TensorBoard(log_dir="./train_log/"), EarlyStopping(patience=10)])

        t = time.time() - t

        import matplotlib.pyplot as plt
        # 損失の履歴をプロット
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='lower right')
        plt.show()

        print("学習終了! 経過時間: {0:.2f}s".format(t))

        # W保存
        name = pyautogui.prompt(text="weight保存名を指定してください", title="AnoVAE",
                                default="ts{0}_zd{1}_b{2}_lam{3}".format(G.TIMESTEPS, G.Z_DIM, G.BATCH_SIZE,
                                                                         G.Loss_Lambda))

        weight_path = "./data/weight/{0}.h5".format(name)
        self.vae.save_weights(filepath=weight_path)
        print("weightを保存しました:\n{0}", weight_path)

        print("Train終了")
        self.load_weight_flag = True
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

    # マルチスレッドでPredict
    def ThreadPredict(self, input_data, thread_size):

        import threading

        X_true, X_true2 = input_data

        # 学習データを分割
        train_datas = []
        for split_data1, split_data2 in zip(np.array_split(X_true, thread_size, axis=0),
                                            np.array_split(X_true2, thread_size, axis=0)):
            train_datas.append([split_data1, split_data2])

        # スレッド内の処理結果を格納する変数
        results = [[] for _ in range(thread_size)]

        # 別スレッドで実行する関数
        def th_func(datas, results,id):

            tf_X_true1, tf_X_true2 = datas

            tf_mu_list, tf_sigma_list, t_z_list = self.encoder.predict(tf_X_true1)

            tf_X_reco = self.decoder.predict([tf_X_true2, t_z_list])
            tf_X_reco = np.reshape(tf_X_reco, newshape=(tf_X_true1.shape[0], G.TIMESTEPS))
            results[id] = [tf_mu_list, tf_sigma_list, tf_X_reco]

        # スレッドのリスト
        th_list = [threading.Thread(target=th_func, args=[train_datas[id], results,id]) for id in range(thread_size)]

        # スレッド処理開始
        for th in th_list:
            th.start()

        start_time = time.time()

        # すべてのスレッドが終了するまで待機
        for th in th_list:
            th.join()

        end_time = time.time()

        mu_list = sigma_list = np.empty(shape=(0, G.Z_DIM))
        X_reco = np.empty(shape=(0, G.TIMESTEPS))
        for r in results:
            mu_list = np.concatenate([mu_list, r[0]], axis=0)
            sigma_list = np.concatenate([sigma_list, r[1]], axis=0)
            X_reco = np.concatenate([X_reco, r[2]], axis=0)

        pro_time = end_time - start_time
        print("再構成完了! スレッド数: {0} 処理時間: {1:.2f}  処理速度: {2:.2f} process/s".format(thread_size, pro_time,
                                                                                X_true.shape[0] / pro_time))
        return mu_list, sigma_list, X_reco

    # 再構成後の再構成後のマンハッタン距離
    def GetReconstructionError(self, X_true, X_reco):
        from scipy.spatial import distance

        re = []
        for x_true, x_reco in zip(X_true, X_reco):
            x_true = np.reshape(x_true, newshape=(-1,))
            re.append(distance.cityblock(x_true, x_reco))

        return re

    # D_KL
    def GetKullbackLeiblerDivergence(self, mu_list, sigma_list):
        dkl = []
        for mu, sigma in zip(mu_list, sigma_list):

            S = 0
            for m, s in zip(mu, sigma):
                S += -0.5 * (1 + s - np.square(m) - np.exp(s))

            dkl.append(S)

        return dkl

    def TestCSV(self, path=None):
        # Wの読み込み
        if not self.load_weight_flag:
            self.LoadWeight()

        # テスト用csvファイルのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "testデータを選んでください")
            path = GetFilePathFromDialog([("テスト用csv", "*.csv"), ("すべてのファイル", "*")])

        # テストデータセット作成
        X_true1, X_true2 = self.BuildData(path, self.MIN_OF_12bit, self.MAX_OF_12bit)

        print("テストデータを読み込みました:\n{0}".format(path))

        print("再構成しています...")

        mu_list, sigma_list, X_reco = self.ThreadPredict([X_true1, X_true2], thread_size=8)

        # reco_view = np.array([])
        # 表示用のX_reco -> reco
        # for x_reco in X_reco[G.TIMESTEPS - 1::G.TIMESTEPS]:
        #    x_reco = np.reshape(x_reco,newshape=G.TIMESTEPS)
        #    reco_view = np.hstack((reco_view, np.reshape(x_reco, newshape=(-1))))

        # ReとD_KLの計算

        # 表示用のX_true
        xt = list(np.reshape(X_true1[0],newshape=(-1,)))
        xt += list(np.reshape(X_true2,newshape=(-1)))

        # 再構成誤差(Re)
        re = [0] * G.TIMESTEPS
        re += self.GetReconstructionError(X_true1, X_reco)

        #元波形におけるμとdecord,re-encordを通したときのμのユークリッド距離
        mm = [0]*int(G.TIMESTEPS/2)
        for mu,re_mu in zip(mu_list[G.TIMESTEPS:],mu_reencord_list[G.TIMESTEPS:]):
            mm.append(distance.euclidean(mu,re_mu))
        dm += [0]*offset

        print("表示用データ作成完了しました")

        return xt, re, dkl


def main():
    vae = AnoVAE()
    if MSGBOX.askyesno("AnoVAE", "AnoVAEに学習させますか？"):
        vae.Train()
    else:
        vae.LoadWeight()

    # for path in glob.iglob("./data/Sin*Test*.csv"):
    #    t,r,e = vae.TestCSV(path)
    #    ShowGlaph(t,r,e)

    t, re, dkl = vae.TestCSV()
    ShowGlaph(t, re, dkl)
    return


if __name__ == "__main__":
    main()
