# VAE関係
# ダイアログ関係
import os
import time
import tkinter
import tkinter.filedialog
import tkinter.messagebox as MSGBOX

import matplotlib.pyplot as plt
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

def GetFilePathFromDialog(file_types):
    # ファイル選択ダイアログの表示
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes=file_types, initialdir=iDir)

    return file

# t_SNEを用いた描画。多分使わない
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
    vae = None      # VAE本体
    encoder = None  # Encoder
    decoder = None  # Decoder

    load_weight_flag = False
    load_minmax_flag = False
    set_threshold_flag = False

    MIN = None
    MAX = None

    THRESHOLD_EG = 0

    # コンストラクタ
    def __init__(self):
        self.vae, self.encoder, self.decoder = self.BuildVAE()
        return

    # データセットをCSVから作成する関数
    def BuildData(self, path):

        if not self.load_minmax_flag:
            self.LoadMINMAX()

        # データ読み込み(サンプル数,)
        X = np.loadtxt(path, encoding="utf-8-sig")

        # 最小値を0にして0-1に圧縮(clampはしない)
        # clamp = lambda x, min_val_a, max_val_a: min(max_val_a, max(x, min_val_a))
        X = np.array(list(map(lambda x: (x - self.MIN) / (self.MAX - self.MIN), X)))

        # 一次元配列から二次元行列に変換(None, 1)
        X = np.reshape(X, newshape=(-1))

        # 全サンプル数(入力csvのデータ数)
        sample_size = X.shape[0] - G.TIMESTEPS + 1

        # X_encoder: encoderに入れるデータセット
        X_encoder = np.zeros(shape=(sample_size, G.TIMESTEPS))

        # X_encoderの作成
        # timestep分スライスして格納
        for i in range(sample_size):
            X_encoder[i] = X[i:i + G.TIMESTEPS]

        # kerasに渡す形(sample,timestep,features)に変換
        X_encoder = np.expand_dims(X_encoder, axis=2)

        return X_encoder, X[G.TIMESTEPS - 1 :]

    # ネットワーク作成
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

        # (None, Z_DIM) <- σ^ (標準偏差σ = exp(log(σ^/2)))
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

        # (None, 1) <-初期値
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

    # 学習
    def Train(self, path=None):

        # 学習データcsvファイルのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE > Train", "学習データを選んでください")
            path = GetFilePathFromDialog([("学習データcsv", "*.csv"), ("すべてのファイル", "*")])
            self.LoadMINMAX(path)

        self.SetMINMAX(0,4095)
        # 学習データを作成
        encoder_inputs, decoder_inputs = self.BuildData(path)
        print("Trainデータ読み込み完了\n{0}".format(path))

        # 学習
        t = time.time()

        from keras.callbacks import TensorBoard, EarlyStopping
        history = self.vae.fit([encoder_inputs, decoder_inputs],
                               epochs=1000,
                               batch_size=G.BATCH_SIZE,
                               shuffle=True,
                               validation_split=0.1,
                               callbacks=[TensorBoard(log_dir="./train_log/"), EarlyStopping(patience=7)])

        t = time.time() - t

        # 損失の履歴をプロット
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.show()

        print("学習終了! 経過時間: {0:.2f}s".format(t))

        # weight保存
        name = pyautogui.prompt(text="weight保存名を指定してください", title="AnoVAE>Train",
                                default="ts{0}_zd{1}_b{2}_lam{3}".format(G.TIMESTEPS, G.Z_DIM, G.BATCH_SIZE,
                                                                         G.Loss_Lambda))

        weight_path = "./data/weight/{0}.h5".format(name)
        self.vae.save_weights(filepath=weight_path)
        print("weightを保存しました:\n{0}", weight_path)

        # ER,EPのしきい値を計算
        self.SetEGThreshold(path)

        print("Train終了")
        self.load_weight_flag = True
        return

    def SetMINMAX(self, MIN, MAX):
        self.MIN = MIN
        self.MAX = MAX
        self.load_minmax_flag = True

        print("SetMINMAX")
        return

    def LoadMINMAX(self, path=None):
        if path is None:
            MSGBOX.showinfo("AnoVAE>LoadMINMAX", "学習で使用したデータを選んでください")
            path = GetFilePathFromDialog([("学習データ", "*.csv"), ("すべてのファイル", "*")])

        X = np.loadtxt(path, encoding="utf-8-sig")
        self.MIN = X.min()
        self.MAX = X.max()
        self.load_minmax_flag = True

        return

    def LoadWeight(self, path=None):

        # weightのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "weightデータを選んでください")
            path = GetFilePathFromDialog([("weight", "*.h5"), ("すべてのファイル", "*")])

        # weightの読み込み
        self.vae.load_weights(path)
        print("weightを読み込みました:\n{0}".format(path))

        self.load_weight_flag = True
        return

    def SetEGThreshold(self, path=None):

        # 正常データのパス
        if path is None:
            MSGBOX.showinfo("AnoVAE>SetThreshould", "正常データを選んでください")
            path = GetFilePathFromDialog([("csv", "*.csv"), ("すべてのファイル", "*")])

        X_encoder, X_decoder = self.BuildData(path)

        mu_list, sigma_list, X_reco = self.ThreadPredict([X_encoder, X_decoder], thread_size=8)

        _, _, eg= self.GetScore(X_encoder, X_reco, mu_list, sigma_list)

        self.THRESHOLD_EG = np.max(eg)

        self.set_threshold_flag = True
        return

    # マルチスレッドでPredict
    def ThreadPredict(self, input_data, thread_size):

        import threading

        X_encoder, X_decoder = input_data

        # 学習データを分割
        train_datas = []
        for split_data1, split_data2 in zip(np.array_split(X_encoder, thread_size, axis=0),
                                            np.array_split(X_decoder, thread_size, axis=0)):
            train_datas.append([split_data1, split_data2])

        # スレッド内の処理結果を格納する変数
        results = [[] for _ in range(thread_size)]

        # 別スレッドで実行する関数
        def th_func(datas, results, id):

            tf_X_true1, tf_X_true2 = datas

            tf_mu_list, tf_sigma_list, t_z_list = self.encoder.predict(tf_X_true1)

            tf_X_reco = self.decoder.predict([tf_X_true2, t_z_list])
            tf_X_reco = np.reshape(tf_X_reco, newshape=(tf_X_true1.shape[0], G.TIMESTEPS))
            results[id] = [tf_mu_list, tf_sigma_list, tf_X_reco]

        # スレッドのリスト
        th_list = [threading.Thread(target=th_func, args=[train_datas[id], results, id]) for id in range(thread_size)]

        # スレッド処理開始
        for th in th_list:
            th.start()

        # すべてのスレッドが終了するまで待機
        for th in th_list:
            th.join()

        # 結果を集積させる
        mu_list = np.empty(shape=(0, G.Z_DIM))
        sigma_list = np.empty(shape=(0, G.Z_DIM))
        X_reco = np.empty(shape=(0, G.TIMESTEPS))
        for r in results:
            mu_list = np.concatenate([mu_list, r[0]], axis=0)
            sigma_list = np.concatenate([sigma_list, r[1]], axis=0)
            X_reco = np.concatenate([X_reco, r[2]], axis=0)

        return mu_list, sigma_list, X_reco

    # 再構成後の再構成後のマンハッタン距離
    def GetReconstructionError(self, X_true, X_reco):
        from scipy.spatial import distance

        re = []
        for x_true, x_reco in zip(X_true, X_reco):
            x_true = np.reshape(x_true, newshape=(-1,))
            re.append(distance.cityblock(x_true, x_reco)/G.TIMESTEPS)

        return re

    # D_KL（ボツ）：計算する意味がなかった
    def GetKullbackLeiblerDivergence(self, mu_list, sigma_list):
        dkl = []
        for mu, sigma in zip(mu_list, sigma_list):

            S = 0
            for m, s in zip(mu, sigma):
                S += -0.5 * (1 + s - np.square(m) - np.exp(s))

            dkl.append(S)

        return dkl

    # 原点からμのユークリッド距離（ボツ）：意味なくはないけど使えるの？
    def GetMuDistance(self, mu_list):
        from scipy.spatial import distance
        md = []
        O = np.zeros(G.Z_DIM)
        for mu in mu_list:
            md.append(distance.euclidean(O, mu))

        return md

    # zを生成する前のN(mu,sigma)が、標準正規分布のkσ区間内[-k,k]になりうる確率
    # zの次元数だけ互いに独立した正規分布 N(μ0,σ0), N(μ1,σ1), ...があるため、
    # すべての事象が起こる確率を計算する
    # この確率が高い→正常である可能性が高い
    def GetSigmaScore(self, k, mu_list, sigma_list):

        #   upper
        # ∫     N(mu,sgm) の計算
        #   lower
        def Prob(lower, upper, mu, sgm):

            import scipy
            idx_l = (lower - mu) / np.sqrt(2) / sgm
            idx_u = (upper - mu) / np.sqrt(2) / sgm

            return 0.5 * (scipy.special.erf(idx_u) - scipy.special.erf(idx_l))

        ss = []

        for mu, sigma in zip(mu_list, sigma_list):

            p = 1.0
            for m, s in zip(mu, sigma):
                p *= Prob(-k, k, m, s)
            ss.append(1 - p)

        return ss

    # 正常データと再構成データから異常度を表すパラメータ(ER,EP,?)を取得する関数
    def GetScore(self, X_true, X_reco, mu_list, sigma_list):

        # 再構成誤差(ER)
        er = self.GetReconstructionError(X_true, X_reco)

        # 分布の計算(EP)
        ep = self.GetSigmaScore(3, mu_list, np.exp(sigma_list / 2))

        # 異常度(Error Rate)

        #算術平均
        def ArithmeticMean(a,b):
            return (a+b)/2
        #相乗平均
        def GeometricMean(a,b):
            return np.sqrt(a*b)
        #調和平均
        def HarmonicMean(a,b):
            if a == 0 and b == 0:
                return 0
            return 2 * a * b /(a + b)

        G_mean = [GeometricMean(R,P) for R,P in zip(er,ep)]

        from scipy.signal import savgol_filter
        eg = list(savgol_filter(G_mean,window_length=21,polyorder=7))

        return er, ep, eg


    def GetBestProminence(self,eg):
        from scipy.optimize import minimize_scalar,minimize
        from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score,accuracy_score

        MSGBOX.showinfo("AnoVAE>TestCSV()", "異常範囲データを指定してください")
        tf_path = GetFilePathFromDialog([("異常範囲データ.csv", "*.csv"), ("すべてのファイル", "*")])

        true = np.loadtxt(tf_path, dtype=bool, encoding="utf-8-sig")  # Ground truth
        true = true[G.TIMESTEPS-1:]

        """
        #最適化する関数
        def Loss(prominence):
            pred,_ = self.FindPeaks(eg,prominence_low=0,prominence_high=prominence)

            #混合行列
            cm = confusion_matrix(true, pred)
            tn, fp, fn, tp = cm.flatten()
            if fp + tp == 0:return 1 #エラー処理

            # recall: 検出率(実際の異常範囲の内、異常と検出できた割合)
            # precision: 精度(予測した異常範囲の内、実際に異常であった割合)

            # 出力はF値(recallとprecisionの調和平均)
            # F値の最大化したいが、minimizeなのでF値の最大値1から減算
            print(prominence)
            return 1 - f1_score(true, pred)

        #bp = minimize_scalar(Loss,bounds=(0.0,max(eg)),method="bounded")
        """

        optimize_low = []
        optimize_high = []
        def Loss2(p_low,p_high):
            pred,_ = self.GetErrorRegion(eg,prominence_low=p_low,prominence_high=p_high)

            #混合行列
            cm = confusion_matrix(true, pred)
            tn, fp, fn, tp = cm.flatten()
            if fp + tp == 0:return 1 #エラー処理

            # recall: 検出率(実際の異常範囲の内、異常と検出できた割合)
            # precision: 精度(予測した異常範囲の内、実際に異常であった割合)

            # 出力はF値(recallとprecisionの調和平均)
            # F値の最大化したいが、minimizeなのでF値の最大値1から減算
            optimize_low.append(p_low)
            optimize_high.append(p_high)

            return 1 - f1_score(true, pred)

        eg_max = max(eg)

        #制約条件cons (x[0] == low,x[1] == high),
        # 0 ≦ low ≦ high ≦ max(eg)
        # "ineq"は不等式 「0 ≦ f(x)」、"fun"は唯のfunctionを表す(偏導関数を与える場合に"jac"と書くが、COBYLAでは使わない)
        cons = ({"type":"ineq","fun":lambda x: x[0]},             # 0    ≦ low
                {"type":"ineq","fun":lambda x: x[1] - x[0]} ,     # low  ≦ high
                {"type":"ineq","fun":lambda x: eg_max - x[1]})    # high ≦ max(eg)

        #探索初期値x0
        x0 = np.array([eg_max,eg_max])

        bp2 = minimize(Loss2,x0=x0,method="COBYLA",constraints=cons)

        max_eg = max(eg)
        div = 100
        x_axis = np.arange(0,max_eg,max_eg/div)
        y_axis = np.arange(0,max_eg,max_eg/div)
        X,Y = np.meshgrid(x_axis,y_axis)

        Z = np.zeros(shape=(div,div))

        for i in range(div):
            for j in range(div):
                low = X[i][j]
                high = Y[i][j]

                if high <= low:
                    Z[i][j] = 0
                    continue
                pred, _ = self.GetErrorRegion(eg, prominence_low=low, prominence_high=high)

                # 混合行列
                cm = confusion_matrix(true, pred)
                tn, fp, fn, tp = cm.flatten()
                if fp + tp == 0: continue  # エラー処理

                print("({0},{1})".format(i,j))
                Z[i][j] = f1_score(true, pred)


        #plt.imshow(Z,interpolation="nearest",cmap="jet")
        cont = plt.contour(X, Y, Z)
        cont.clabel(fmt="%1.1f",fontsize=14)

        for i in range(len(optimize_low)-1):
            plt.anotate("",
                        xy=(optimize_low[i+1],optimize_high[i+1]),
                        xytext=(optimize_low[i],optimize_high[i]),
                        arrowstyle="->"
                        )

        plt.xlabel("prominence low")
        plt.ylabel("prominence high")
        plt.show()

        return bp2.x[0],bp2.x[1]

    def FindPeaks(self,eg,prominence_low,prominence_high):
        from scipy.signal import find_peaks,peak_prominences
        pred = [False] * len(eg)
        h = self.THRESHOLD_EG  # 最低ピーク値
        #d = int(G.TIMESTEPS * 0.5)  # ピーク同士の距離の最小値
        wlen = G.TIMESTEPS

        # ピーク検出
        peaks,properties = find_peaks(eg, height=h,wlen=wlen,prominence=prominence_low)
        # Error特定(ピークの左端を利用する), eg[peak] - eg[l_base] と prominence(最適化対象)を比較
        peak_x_list = []
        l_index_list = []
        r_index_list = []
        for peak, l_base,r_base in zip(peaks, properties["left_bases"],properties["right_bases"]):
            if eg[l_base] > eg[r_base]:continue
            if eg[peak] - eg[l_base] < prominence_high:continue

            peak_x_list.append(peak)
            l_index_list.append(l_base)
            r_index_list.append(r_base)
            for i in range(l_base, r_base + 1):
                pred[i] = True

        #l_baseがかぶってる要素を消去
        remove_list = []
        for i in range(1,len(l_index_list)):
            if l_index_list[i] == l_index_list[i-1]:
                remove_list.append(i-1)
        remove_list = remove_list[::-1]
        for i in remove_list:
            del peak_x_list[i]
            del l_index_list[i]
            del r_index_list[i]

        return pred,[peak_x_list,l_index_list,r_index_list]

    def GetErrorRegion(self,eg,prominence_low,prominence_high):
        pred, peaks_data = self.FindPeaks(eg, prominence_low=prominence_low,prominence_high=prominence_high)
        pred = [P and T for P,T in zip(pred,np.array(eg) > self.THRESHOLD_EG)]
        return pred,peaks_data


    def GetErrorRateThreshold(self, error_rate):
        from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score,accuracy_score

        MSGBOX.showinfo("AnoVAE>TestCSV()", "異常範囲データを指定してください")
        tf_path = GetFilePathFromDialog([("異常範囲データ.csv", "*.csv"), ("すべてのファイル", "*")])

        true = np.loadtxt(tf_path, dtype=bool, encoding="utf-8-sig")  # 真値Ground truth
        error_rate_np = np.array(error_rate)  # 予測値

        recall_list = []
        precision_list = []
        F_list = []
        max_F = 0
        max_threshold = 0
        accuracy = 0
        error_rate_max = max(error_rate)
        for threshold in range(G.TIMESTEPS):
            threshold *= error_rate_max / G.TIMESTEPS
            pred = error_rate_np >= threshold

            cm = confusion_matrix(true, pred)
            tn, fp, fn, tp = cm.flatten()
            if fp + tp == 0:
                recall_list.append(None)
                precision_list.append(None)
                F_list.append(None)
                continue

            recall_list.append(recall_score(true, pred))  # 検出率
            precision_list.append(precision_score(true, pred))  # 精度
            F = f1_score(true, pred)
            if F > max_F:
                max_F = F
                max_threshold = threshold
                accuracy = accuracy_score(true,pred)

            F_list.append(F)

        print("accuracy = {0}".format(accuracy))
        # グラフ
        plt.ylabel("")
        plt.ylim(0, 1)
        x_axis = range(len(recall_list))
        plt.plot(x_axis, recall_list, label="Recall")
        plt.plot(x_axis, precision_list, label="Precision")
        plt.plot(x_axis, F_list, label="F-score")
        plt.legend()

        plt.show()

        return max_threshold

    def ShowScoreGlaph(self, true, er, ep,eg):

        x_axis = range(len(true))
        offset = [0] * (G.TIMESTEPS-1) # ER,EP,G表示用

        # original
        plt.subplot(4, 1, 1)
        plt.ylabel("Value")
        plt.ylim(0, 1)
        plt.plot(x_axis, true, label="original")
        plt.legend()

        # Reconstruction Error
        plt.subplot(4, 1, 2)
        plt.ylabel("ER")
        #plt.ylim(0, 1)
        plt.plot(x_axis, offset + er, label="Reconstruction Error")
        plt.legend()

        # Probability Error
        plt.subplot(4, 1, 3)
        plt.ylabel("EP")
        #plt.ylim(0, 1)
        plt.plot(x_axis, offset + ep, label="Probability Error")
        plt.legend()

        # EG
        plt.subplot(4, 1, 4)
        plt.ylabel("EG")
        #plt.ylim(0, 1)
        plt.plot(x_axis, offset + eg, label="Geometric mean + GS7-21 Filter")
        plt.legend()

        plt.show()

        return

    def ShowErrorRegion(self,true,pred,eg,peaks_data):

        x_axis = range(len(true))

        offset = [0] * (G.TIMESTEPS - 1)
        pred = offset + pred

        # 異常領域の色塗り
        start_flag = False
        start_pos = 0
        error_range = 0

        start_pos_list = []
        end_pos_list = []

        for i in range(len(pred)):

            if start_flag:
                if pred[i]:
                    error_range += 1
                    continue
                start_pos_list.append(start_pos - 0.5)
                end_pos_list.append(start_pos + error_range + 0.5)
                error_range = 0
                start_flag = False

            if pred[i]:
                start_flag = True
                start_pos = i
        else:
            if start_flag:
                start_pos_list.append(start_pos - 0.5)
                end_pos_list.append(start_pos + error_range + 0.5)

        # original
        plt.subplot(2, 1, 1)
        plt.ylabel("Value")
        plt.ylim(0, 1)

        for start,end in zip(start_pos_list,end_pos_list):
            plt.axvspan(start,end, color="#ffcdd2")
        plt.plot(x_axis, true, label="original")
        plt.legend()

        # EG
        plt.subplot(2, 1, 2)
        plt.ylabel("EG")

        eg = offset + eg

        # index
        peaks = [peak + len(offset) for peak in peaks_data[0]]
        l_bases = [lb + len(offset) for lb in peaks_data[1]]
        r_bases = [rb + len(offset) for rb in peaks_data[2]]

        # eg[index]
        eg_peaks = [eg[peak] for peak in peaks]
        eg_l_bases = [eg[lb] for lb in l_bases]
        eg_r_bases = [eg[rb] for rb in r_bases]

        # plot
        from matplotlib.markers import CARETDOWN
        plt.plot(peaks, eg_peaks, marker=CARETDOWN, markersize=10, color="red",label="peak",linestyle="None")             # ▼ピーク位置
        plt.vlines(peaks, ymin=eg_r_bases, ymax=eg_peaks, color="orange",label="prominence",lw=2)                        # prominence
        plt.vlines(peaks, ymin=eg_l_bases, ymax=eg_r_bases, color="orange",linestyles="--",lw=2)                       # l_base 〜 r_base
        plt.hlines(eg_l_bases, xmin=l_bases, xmax=peaks, color="green",lw=2)   # l_base
        plt.hlines(eg_r_bases, xmin=peaks, xmax=r_bases, color="lime",lw=2)   # r_base

        plt.plot(x_axis, eg, label="EG")
        plt.hlines([self.THRESHOLD_EG for _ in range(len(true))], xmin=0, xmax=len(true), color="purple",linestyles="--", lw=2,label="threshold")  # threshold
        plt.xlabel("time")
        plt.legend()

        plt.show()

    # テストデータ(CSV)を評価する関数
    def TestCSV(self, path=None):

        print("CSVTestを実行します")

        ############################# パラメータの設定 ##############################
        # weightの読み込み
        if not self.load_weight_flag:
            self.LoadWeight()
        print("重みデータを読み込みました")

        # 閾値の読み込み
        if not self.set_threshold_flag:
            self.SetEGThreshold()
        print("評価指標用の閾値の設定を行いました\n EG:{0}".format(self.THRESHOLD_EG))

        # minmaxの設定
        if not self.load_minmax_flag:
            self.LoadMINMAX()
        print("学習レンジの設定を行いました\n min:{0}   MAX:{1}".format(self.MIN, self.MAX))

        ############################# 推論 ##############################

        # テスト用csvファイルのパスを取得
        if path is None:
            MSGBOX.showinfo("AnoVAE", "testデータを選んでください")
            path = GetFilePathFromDialog([("テスト用csv", "*.csv"), ("すべてのファイル", "*")])

        # テストデータセット作成
        X_encoder, X_decoder = self.BuildData(path)
        print("データセットを作成しました:\n{0}".format(path))
        print("再構成しています...")

        # 再構成
        t = time.time()
        mu_list, sigma_list, X_reco = self.ThreadPredict([X_encoder, X_decoder], thread_size=8)
        pro_time = time.time() - t
        print("再構成完了! 処理時間: {0:.2f}s  処理速度: {1:.2f} process/s".format(pro_time, X_encoder.shape[0] / pro_time))

        ############################# 評価 ##############################

        # 表示用のX_true
        t = time.time()
        true = list(np.reshape(X_encoder[0], newshape=(-1,)))
        true += [X_encoder[i][G.TIMESTEPS - 1][0] for i in range(1, X_encoder.shape[0])]

        # 評価指標計算
        er, ep,eg = self.GetScore(X_encoder, X_reco, mu_list, sigma_list)

        self.ShowScoreGlaph(true, er, ep,eg)
        print("表示用データ作成完了しました 処理時間: {0:.2f}s".format(time.time() - t))

        # 閾値決定
        #error_threshold = self.GetErrorRateThreshold(error_rate)

        best_p_low,best_p_high = self.GetBestProminence(eg)
        pred,peaks_data = self.GetErrorRegion(eg,prominence_low=best_p_low,prominence_high=best_p_high)

        self.ShowErrorRegion(true,pred,eg,peaks_data)

        #self.ShowErrorRegion(true, error_rate, error_threshold)

        ############################# 2回目 推論 ##############################

        # テスト用csvファイルのパスを取得
        MSGBOX.showinfo("AnoVAE", "testデータを選んでください")
        path = GetFilePathFromDialog([("テスト用csv", "*.csv"), ("すべてのファイル", "*")])

        # テストデータセット作成
        X_encoder, X_decoder = self.BuildData(path)

        # 再構成
        mu_list, sigma_list, X_reco = self.ThreadPredict([X_encoder, X_decoder], thread_size=8)

        ############################# 評価 ##############################

        # 表示用のX_true
        true = list(np.reshape(X_encoder[0], newshape=(-1,)))
        true += [X_encoder[i][G.TIMESTEPS - 1][0] for i in range(1, X_encoder.shape[0])]

        # 評価指標計算
        _, _, eg = self.GetScore(X_encoder, X_reco, mu_list, sigma_list)
        pred,peaks_data = self.GetErrorRegion(eg,prominence_low=best_p_low,prominence_high=best_p_high)

        self.ShowErrorRegion(true, pred,eg,peaks_data)

        return


def main():
    vae = AnoVAE()
    vae.SetMINMAX(0, 4095)

    if MSGBOX.askyesno("AnoVAE", "AnoVAEに学習させますか？"):
        vae.Train()
    else:
        vae.LoadWeight()
        vae.SetEGThreshold()

    vae.TestCSV()
    return


if __name__ == "__main__":
    main()
