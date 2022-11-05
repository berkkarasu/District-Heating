from itertools import product
from mip import Model, BINARY, INTEGER, xsum
import sys
sys.path.append("C:\\Users\\admin\\Desktop\\Ozyegin\\District Heating")
import pandas as pd
import numpy as np
import joblib
import csv
import time

start = time.time()
model = Model('SDHP')
df = joblib.load('paper_synth_data.pk')
results = dict()
k = (3,6)
v = df[k]
J = k[0]
T=int(k[1]*60/5)+1
 
K_int= v['init_temp']
J=len(K_int)
D= v['demanded_temp']

D= np.repeat(D,12,1)
D = np.concatenate([np.array(K_int).reshape(-1,1),D],1)
##PARAMETERS
L=3
I=4

#cost of energy at time t, c_l
c = [5,15,40]
N=[0.0166, 0.0630, 0.0824, 0.1046]

p = [1/6,1/2,1]
Q=[]
for l in range(L):
    Q.append([J*N[-1]*p[l]]*T)

#epsilon
eps=0.01
#c_ap*m_j[j] humidity 43-45
h=44
#c_wp*m_i[i]*(T_sup[j]-T_out[j]) a_ij
coefs=[0, 15.095, 15.49728, 15.89319]
a=np.array(J*[N]).T

#Rij c_wp*m_i[i]*(T_sup[j]-T_out[j])/(c_ap*m_j[j])
R=np.array(J*[coefs]).T/h

#-L_p[j][t] + G[j][t]/c_ap*m_j[j] Bj intercept her daire için aynı
b = np.array(J*[-.175469])/h

X = [[model.add_var(name='X({},{})'.format(j, t)) for t in range(T)] for j in range (J)]

K = [[model.add_var(name='K({},{})'.format(j, t)) for t in range(T)] for j in range (J)]

S = [[model.add_var(name='S({},{})'.format(j, t)) for t in range(T)]for j in range (J)]

Z = [[[model.add_var(var_type = BINARY, name='Z({},{},{})'.format(j, t, i)) for i in range(I)] for t in range (T)] for j in range(J)]

U = [[model.add_var(name='U({},{})'.format(j, t)) for t in range(T)] for j in range (J)]

Y = [[model.add_var(name='Y({},{})'.format(l, t)) for t in range(T)] for l in range (L)]

model.objective = xsum(c[l]*Y[l][t] for l in range (L) for t in range(T-1))+xsum((100*U[j][t]) for j in range(J) for t in range(T-1))

for j in range(J):
         for t in range(1,T):
                model += K[j][t]+U[j][t] >= D[j][t]

for j in range(J):
        model += K[j][0] == K_int[j]

for j in range(J):
        for t in range(T-1):
                model += K[j][t+1]-K[j][t] ==\
              (xsum(R[i][j]*Z[j][t][i] for i in range(I))) + b[j]

for i in range(I):
        for j in range(J):
                for t in range(T-1):
                        model+= X[j][t] == xsum(a[i][j]*Z[j][t][i] for i in range(I))

for j in range(J):
        for t in range(T-1):
                model += S[j][t]-K[j][t] <= 0.1 + 30*(1-Z[j][t][0])

for j in range(J):
        for t in range(T-1):
                model += 0.1 - 30*(1-Z[j][t][1]) + eps <= S[j][t]-K[j][t]
        
for j in range(J):
        for t in range(T-1):
                model += S[j][t]-K[j][t] <= 0.5 + 30*(1-Z[j][t][1])

for j in range(J):
        for t in range(T-1):
                model += 0.5 - 30*(1-Z[j][t][2]) + eps <= S[j][t]-K[j][t]

for j in range(J):
        for t in range(T-1):
                model += S[j][t]-K[j][t] <= 1.5 + 30*(1-Z[j][t][2])

for j in range(J):
        for t in range(T-1):
                model += 1.5 - 30*(1-Z[j][t][3]) + eps <= S[j][t]-K[j][t]

for j in range(J):
        for t in range(T-1):
                model += S[j][t]-K[j][t] <= 10 + 30*(1-Z[j][t][3])

for j in range(J):
        for t in range(T-1):
                model += Z[j][t][0] + Z[j][t][1] + Z[j][t][2] + Z[j][t][3] == 1

for t in range(T-1):
        model += xsum(Y[l][t] for l in range(L)) == 1


for t in range(T-1):
        model += xsum(X[j][t] for j in range(J)) <=\
        xsum(Q[l][t]*Y[l][t] for l in range(L))

model.optimize(max_seconds = 1200)
print('')
print('Objective value: {model.objective_value:.3}'.format(**locals()))
print('Solution: ', end='')
for v in model.vars:
    if v.x > 1e-5:
        print('{v.name} = {v.x}'.format(**locals()))
        print('          ', end='')