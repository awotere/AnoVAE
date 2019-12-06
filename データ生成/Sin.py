
import csv
import numpy as np
import sys

import データ生成.AnomalyDomain as AD
import random

RANGE = 5000 #生成レンジ

#設定
T = 1000                 #周期
A = 1024                 #振幅 (必ず1024未満にするべし)
domain_flag = False     #ドメインを使うか
#/設定

array = [0]*RANGE
domain = AD.AnomalyDomain(2,1,0,1)

d_str = ""
if domain_flag:
    d_str = "_TestDomain"

#file = open("./data/SinAno(T{0}_A{1}{2}).csv".format(T,A,d_str),"w",newline="")
file = open("./random_.csv","w",newline="")
writer = csv.writer(file)



def RandomDomain():
    if domain.UpdateCheck():
        new_dom_range = int(random.uniform(100,600))
        new_ano_start = int(random.uniform(0,new_dom_range))
        new_ano_range = int(random.uniform(10,min(200,new_dom_range)))
        new_ano_scale = random.uniform(0.5,2)
        domain.UpdateDomain(new_dom_range,new_ano_start,new_ano_range,new_ano_scale)

    return int(domain.GetValue(t) * A)

def TestDomain():
    if domain.UpdateCheck():

        ano_range = 50
        start_pos = 150 - ano_range / 2

        new_dom_range = int(500)
        new_ano_start = int(start_pos)
        new_ano_range = int(ano_range)
        new_ano_scale = TestDomain.scale_list[TestDomain.phase]
        domain.UpdateDomain(new_dom_range,new_ano_start,new_ano_range,new_ano_scale)

        TestDomain.phase += 1

    return int(domain.GetValue(t) * A)
TestDomain.phase = 0
TestDomain.range_list = [10,10,20,20,50,50,100,100,200,200]
TestDomain.scale_list = [0.25,0.25,0.5,0.5,0.75,0.75,1,1,1.25,1.25]

for t in range(RANGE):
    #sin: y = int(np.sin( t / T * 2 * np.pi) * A + 2047)
    y = int(500*random.uniform(-1,1))+2047

    if domain_flag:
        y += RandomDomain()

    writer.writerow([y])

print("おわり")
file.close()
