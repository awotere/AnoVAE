import numpy as np

class AnomalyDomain:
    
    #変数
    DOMAIN_RANGE = 200
    ANOMALY_START = 150
    ANOMALY_RANGE = 20
    ANOMALY_SCALE = 1
    _count = 0 #内部カウント

    array = [0]*DOMAIN_RANGE

    def __init__(self,d_range = DOMAIN_RANGE,a_start = ANOMALY_START,a_range = ANOMALY_RANGE,a_scale = ANOMALY_SCALE):    
        self.UpdateDomain(d_range,a_start,a_range,a_scale)
        return

    def UpdateCheck(self):
        if self._count >= self.DOMAIN_RANGE:
            self._count = 0
            return True
        return False


    def UpdateDomain(self,dom_range,ano_start,ano_range,ano_scale):
        self.DOMAIN_RANGE = dom_range
        self.ANOMALY_RANGE = ano_range
        self.ANOMALY_SCALE = ano_scale

        x = ano_start
        while x >= dom_range - ano_range:
            x -= (dom_range - ano_range)
        self.ANOMALY_START = x

        self.array = [0]*self.DOMAIN_RANGE
        for t in range(self.ANOMALY_RANGE):
            self.array[self.ANOMALY_START+t] += (np.cos(t / self.ANOMALY_RANGE * 2*np.pi) - 1) / 2 * self.ANOMALY_SCALE
        return

    def GetValue(self,t):
        while t >= self.DOMAIN_RANGE:
            t -= self.DOMAIN_RANGE

        self._count += 1
        return self.array[t]

