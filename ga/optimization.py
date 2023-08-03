# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:12:54 2020

@author: Duygu Ay, Mehmet Berk Karasu
"""

import sys
sys.path.append("C:\\Users\\admin\\Desktop\\Ozyegin\\District Heating")
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import joblib

#data reading

df=joblib.load('16-18-4.pk')
results=dict()
#for k, v in df.items()[1:]:
k=(16,18)
v=df[k]
J=k[0]
T=int(k[1]*60/5)+1

#creating sythetic dataset
"""
import pickle

or2=dict()
or2['init_temp']=np.array(K_int)
or2['demanded_temp']=np.array(D_all)
with open("synth_data_1012.pk", "wb") as tf:
    pickle.dump(or2,tf)

import random
K_int=[19.9, 19.4, 17.8, 17.5, 18.9, 19.3, 20.1, 18.7, 20.3, 18.1]
D_all=[]
T_s=[]
for i in range(len(K_int)):
    for j in range(12):
        if j==0:
            T_s.append(K_int[i]+random.choice(np.arange(-2.5,3.0,0.2)))
        else:
            T_s.append(T_s[j-1]+random.choice(np.arange(-2.5,3.0,0.2)))
    D_all.append(T_s)
    T_s=[]
"""   
 
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
N=np.array([0.0166, 0.0630, 0.0824, 0.1046])

p = [1/6,1/2,1]

Q=[]
for l in range(L):
    Q.append([J*N[-1]*p[l]]*T)

#epsilon
eps=0.03

#c_ap*m_j[j] humidity 43-45
h=44
#c_wp*m_i[i]*(T_sup[j]-T_out[j]) a_ij
coefs=[0, 15.095, 15.49728, 15.89319]
a=np.array(J*[N]).T

#Rij c_wp*m_i[i]*(T_sup[j]-T_out[j])/(c_ap*m_j[j])
R=np.array(J*[coefs]).T/h

#-L_p[j][t] + G[j][t]/c_ap*m_j[j] Bj intercept her daire için aynı
b = np.array(J*[-.175469])/h

##DECISION VARIABLES

m = gp.Model('QUANTCO_Vattenfall')
#m.setParam('TimeLimit', 540*60)
m.setParam('MIPGAP', 0.015)
#the amount of energy used in flat j in period t, X_jt
X = m.addVars(J, T, lb=0.0, vtype=GRB.CONTINUOUS, name='X')

#the temperature of flat j in period t, K_jt
K = m.addVars(J, T, lb=0.0, vtype=GRB.CONTINUOUS, name='K')

#the set temperature of flat j in period t at the smart regulator, S_jt
S = m.addVars(J, T, lb=0.0, vtype=GRB.CONTINUOUS, name='S')

#1 if the heater at flat j operates in mode i in period t, 0 otherwise, Z_jti
Z = m.addVars(J, T, I, vtype=GRB.BINARY, name='Z')

#Error cost of cannot satisfying the termal demand

U = m.addVars(J, T, vtype=GRB.CONTINUOUS, name='U')

#Binary cost of unmet demand
BU = m.addVars(J, T, vtype=GRB.BINARY, name='BU')

Y = m.addVars(L, T, vtype=GRB.BINARY, name='Y')

#Set objective
obj2=gp.quicksum(c[l]*Y[l,t] for l in range(L) for t in range(T-1))+gp.quicksum((10*BU[j,t]+100*U[j,t]) for j in range(J) for t in range(T-1))
m.setObjective(obj2)

#S_opt =

#Add constraints
m.addConstrs((K[j,t]+U[j,t] >= D[j][t]-0.03 for j in range(J) for t in range(1,T-1)),"cons1")
m.addConstrs((U[j,t] <= 0.86*BU[j,t] for j in range(J) for t in range(1,T-1)),"cons1b")
# m.addConstrs((K[j,1]-K_int[j] ==\
#               (gp.quicksum(R[i][j]*Z[j,0,i] for i in range(I))) + b[j] \
#                for j in range(J)),"cons2.a")

m.addConstrs((K[j,0]-U[j,0] == K_int[j] for j in range(J)),"cons12")

m.addConstrs((K[j,t+1]-K[j,t]-U[j,t+1] ==\
              (gp.quicksum(R[i][j]*Z[j,t,i] for i in range(I))) + b[j] \
               for j in range(J) for t in range(T-1)),"cons2")#L_p[j][t] - G[j][t]= 0.25
      
m.addConstrs((X[j,t] ==\
              gp.quicksum(a[i][j]*Z[j,t,i] for i in range(I))\
              for j in range(J) for t in range(T-1)),"cons3")
              #c_wp*(T_sup[j]-T_out[j])*(gp.quicksum(m_i[i]*Z[j,t,i] for i in range(I)))\
              #for j in range(J) for t in range(T)),"cons3")

m.addConstrs((S[j,t]-K[j,t]-U[j,t] <= 0.1 + 30*(1-Z[j,t,0]) for j in range(J) for t in range(T-1)),"cons4")

m.addConstrs((0.1 - 30*(1-Z[j,t,1]) + eps <= S[j,t]-K[j,t]-U[j,t] for j in range(J) for t in range(T-1)),"cons5part1")

m.addConstrs((S[j,t]-K[j,t]-U[j,t] <= 0.5 + 30*(1-Z[j,t,1]) for j in range(J) for t in range(T-1)),"cons5part2")

m.addConstrs((0.5 - 30*(1-Z[j,t,2]) + eps <= S[j,t]-K[j,t]-U[j,t] for j in range(J) for t in range(T-1)),"cons6part1")

m.addConstrs((S[j,t]-K[j,t]-U[j,t] <= 1.5 + 30*(1-Z[j,t,2]) for j in range(J) for t in range(T-1)),"cons6part2")

m.addConstrs((1.5 - 30*(1-Z[j,t,3]) + eps <= S[j,t]-K[j,t]-U[j,t] for j in range(J) for t in range(T-1)),"cons7part1")

m.addConstrs((S[j,t]-K[j,t]-U[j,t] <= 10 + 30*(1-Z[j,t,3]) for j in range(J) for t in range(T-1)),"cons7part2")

#m.addConstrs((18*(1-Z[j,t,0]) <= S[j,t//4] for j in range(J) for t in range(T)),"cons8part1")

#m.addConstrs((S[j,t//4] <= 23*(1-Z[j,t,0]) for j in range(J) for t in range(T)),"cons8part2")

m.addConstrs((Z[j,t,0] + Z[j,t,1] + Z[j,t,2] + Z[j,t,3]==1 for j in range(J) for t in range(T-1)),"cons9")

m.addConstrs((gp.quicksum(Y[l,t] for l in range(L)) == 1 for t in range(T-1)),"cons13")

m.addConstrs((gp.quicksum(X[j,t] for j in range(J)) <=\
               gp.quicksum(Q[l][t]*Y[l,t] for l in range(L)) for t in range(T-1)),"cons10")

#m.addConstrs((E[l,t] <= q[l]*Q[t] for l in range(L) for t in range(T)),"cons11")

#m.addConstrs((S[j,t] == S_opt[j][t] for j in range(J) for t in range(T//4)),"cons13")

# Optimize model
m.optimize()

print('Obj: %g' % m.objVal)
print('Energy consumption cost Obj: %g' % gp.quicksum(c[l]*Y[l,t] for l in range(L) for t in range(T-1)).getValue())
print('Unmet  cost Obj: %g' % gp.quicksum((10*BU[j,t]+100*U[j,t]) for j in range(J) for t in range(T-1)).getValue())
print('Energy consumption %g' % gp.quicksum((X[j,t]) for j in range(J) for t in range(T-1)).getValue())

k=m.getAttr('x', Z)
z_list = np.zeros(shape = (J,T,I))
for j in range(J):
    for t in range(T):
        for i in range(I):
            z_list[j,t,i] = Z[j,t,i].X

unmet_list = np.zeros(shape=(J,T))
for j in range(J):
    for t in range(T):
        unmet_list[j][t] = U[j,t].X

energy_list_flat = np.zeros(shape=(J,T))
for j in range(J):
    for t in range(T):
        energy_list_flat[j][t] = X[j,t].X

energy_list = np.zeros(shape= (T))
energy_list = energy_list_flat.sum(axis=0)

mods = np.zeros(shape=(z_list.shape[0],z_list.shape[1]))
for j in range(z_list.shape[0]):
    for t in range(z_list.shape[1]):
        if z_list[j][t][0] == 1:
            mods[j][t] = 0
        elif z_list[j][t][1] == 1:
            mods[j][t] = 1
        elif z_list[j][t][2] == 1:
            mods[j][t] = 2   
        else:
            mods[j][t] = 3

k=m.getAttr('x', Z)
k_list = np.zeros(shape = (J,T))
for j in range(J):
    for t in range(T):
        k_list[j,t] = K[j,t].X

pd.DataFrame(mods).to_csv("modlist.csv", index_label="Index", header =range(T))
m.dispose()
