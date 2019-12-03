import numpy as np

TIMESTEPS = 200
#INTERMIDIATE_DIM = 100
Z_DIM = 25

BATCH_SIZE = 512

#DATA_MIN = -150
#DATA_MAX = 150

#X_LOG = np.zeros(NUM_TIMESTEPS)
#ERROR_LOG = np.zeros(NUM_TIMESTEPS)

#クランプ関数
def Clamp(x,m,M):
    return min(M, max(x, m))

#TEST_FILE = np.loadtxt("./data/BPF6ch(test_data2).csv",encoding = "utf-8-sig")

