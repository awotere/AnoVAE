from GRU_VAE import BuildVAE,BuildData
import numpy as np

if __name__ == "__main__":
    NUM_TIMESTEPS = 50
    X_test = BuildData("./data/BPF6ch(test_data2).csv",-150,150)
    vae = BuildVAE()
    vae.load_weights("./MNIST_gru_vae/MNIST_gru_vae.h5")

    
    # 推論

    import time

    t = time.time()
    X_pred = vae.predict(X_test)
    print("経過時間:", time.time()-t)

    import matplotlib.pyplot as plt

    array = [0]*50

    for i in range(X_test.shape[0] - NUM_TIMESTEPS):

        true = X_test[i:i+NUM_TIMESTEPS]
        pred = X_pred[i:i+NUM_TIMESTEPS]

        sum = 0
        for j in range(NUM_TIMESTEPS):
            sum += abs(true[j][0] - pred[j][0])
        #print("step = ", i ," Error Rate : ",sum)
        array.append(sum)

    true = [x[0][0] for x in X_test]
    pred = [x[0][0] for x in X_pred]

    plt.subplot(2,1,1)
    plt.ylabel("AD")
    plt.xlabel("time")
    plt.xlim(0,1000)

    plt.plot(range(len(true)),true,label = "original")
    plt.plot(range(len(pred)),pred,label = "reconstructed")
    
    plt.legend()

    plt.subplot(2,1,2)
    plt.ylabel("Error Rate")
    plt.xlabel("time")
    plt.xlim(0,1000)
    plt.plot(range(len(array)),array, label="ErrorRate")

    plt.legend()
    plt.show()