# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:12:54 2020

@author: Duygu Ay
"""

import sys

from matplotlib import test
sys.path.append("C:\\Users\\admin\\Desktop\\Ozyegin\\District Heating")
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import joblib
import csv
import time

#data reading
start = time.time()
df=joblib.load('paper_synth_data.pk')
results=dict()
#for k, v in df.items()[1:]:
k=(3,6)
v=df[k]
J=k[0]
T=int(k[1]*60/5)+1
"""
mods_string = []
mods = []
# Open the file and read the content in a list
with open('mod_lists1_ga.txt', 'r') as filehandle:
    mods_string = [current_place.rstrip() for current_place in filehandle.readlines()]


for i in range(len(mods_string)):
    if mods_string[i].strip().isdigit():
        mods_string[i] = mods_string[i].split(",")

for i in range(len(mods_string)):
    mods.append(mods_string[i].split(","))
modulations = np.zeros(shape=(k[0], T))
for i in range(modulations.shape[0]):
    for j in range(len(mods[0])):
        modulations[i][j+1] = int(mods[i][j][1])
modulations.reshape((k[0], T))

Zs = np.zeros(shape=(k[0], T, 4))
for j in range(k[0]):
    for t in range(T):
        Zs[j,t,int(modulations[j,t])] = 1

print(Zs)
"""
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
N=[0.0166, 0.0630, 0.0824, 0.1046]

p = [1/6,1/2,1]


"""
Q=[0.28,0.52,1.046]
"""
"""
for l in range(L):
    Q.append([J*N[-1]*p[l]]*T)

for l in range(L):
    if l == 0:
        Q.append([J*0.2]*T)
    if l == 1:
        Q.append([J*0.]*T)
    else:
        Q.append([1]*T)

"""

Q=[]
for l in range(L):
    Q.append([J*N[-1]*p[l]]*T)

#epsilon
eps=0.01
epsilon_1 = 0.2
epsilon_2 = 0.2
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
m.setParam('MIPGap', 0.01)
m.setParam('TimeLimit', 30*60)

#the amount of energy used in flat j in period t, X_jt
X = m.addVars(J, T, lb=0.0, vtype=GRB.CONTINUOUS, name='X')

#the temperature of flat j in period t, K_jt
K = m.addVars(J, T, lb=0.0, vtype=GRB.CONTINUOUS, name='K')

#the set temperature of flat j in period t at the smart regulator, S_jt
S = m.addVars(J, T, lb=0.0, vtype=GRB.CONTINUOUS, name='S')

#1 if the heater at flat j operates in mode i in period t, 0 otherwise, Z_jti
Z = m.addVars(J, T, I, vtype=GRB.BINARY, name='Z')

#Error cost of cannot satisfying the termal demand

#U = m.addVars(J, T, vtype=GRB.CONTINUOUS, name='U')

#Binary cost of unmet demand
#BU = m.addVars(J, T, vtype=GRB.CONTINUOUS, name='BU')

Y = m.addVars(L, T, vtype=GRB.BINARY, name='Y')

#Set objective

obj2=gp.quicksum(c[l]*Y[l,t] for l in range(L) for t in range(T-1))#+gp.quicksum((60*U[j,t]) for j in range(J) for t in range(T-1))
m.setObjective(obj2)

#S_opt =

#Add constraints
m.addConstrs((K[j,t]+epsilon_1 >= D[j][t] for j in range(J) for t in range(1,T)),"cons1")

# m.addConstrs((K[j,1]-K_int[j] ==\
#               (gp.quicksum(R[i][j]*Z[j,0,i] for i in range(I))) + b[j] \
#                for j in range(J)),"cons2.a")

m.addConstrs((K[j,0] == K_int[j] for j in range(J)),"cons12")

m.addConstrs((K[j,t+1]-K[j,t] ==\
              (gp.quicksum(R[i][j]*Z[j,t,i] for i in range(I))) + b[j] \
               for j in range(J) for t in range(T-1)),"cons2")#L_p[j][t] - G[j][t]= 0.25

m.addConstrs((X[j,t] ==\
              gp.quicksum(a[i][j]*Z[j,t,i] for i in range(I))\
              for j in range(J) for t in range(T-1)),"cons3")
              #c_wp*(T_su p[j]-T_out[j])*(gp.quicksum(m_i[i]*Z[j,t,i] for i in range(I)))\
              #for j in range(J) for t in range(T)),"cons3")

m.addConstrs((S[j,t]-K[j,t] <= 0.1 + 30*(1-Z[j,t,0])+epsilon_2 for j in range(J) for t in range(T-1)),"cons4")

m.addConstrs((0.1 - 30*(1-Z[j,t,1]) + eps <= S[j,t]-K[j,t]  for j in range(J) for t in range(T-1)),"cons5part1")

m.addConstrs((S[j,t]-K[j,t] <= 0.5 + 30*(1-Z[j,t,1])+epsilon_2 for j in range(J) for t in range(T-1)),"cons5part2")

m.addConstrs((0.5 - 30*(1-Z[j,t,2]) + eps <= S[j,t]-K[j,t]  for j in range(J) for t in range(T-1)),"cons6part1")

m.addConstrs((S[j,t]-K[j,t] <= 1.5 + 30*(1-Z[j,t,2])+epsilon_2 for j in range(J) for t in range(T-1)),"cons6part2")

m.addConstrs((1.5 - 30*(1-Z[j,t,3]) + eps <= S[j,t]-K[j,t] for j in range(J) for t in range(T-1)),"cons7part1")

m.addConstrs((S[j,t]-K[j,t] <= 10 + 30*(1-Z[j,t,3])+epsilon_2  for j in range(J) for t in range(T-1)),"cons7part2")


#m.addConstrs((18*(1-Z[j,t,0]) <= S[j,t//4] for j in range(J) for t in range(T)),"cons8part1")

#m.addConstrs((S[j,t//4] <= 23*(1-Z[j,t,0]) for j in range(J) for t in range(T)),"cons8part2")

m.addConstrs((Z[j,t,0] + Z[j,t,1] + Z[j,t,2] + Z[j,t,3]==1 for j in range(J) for t in range(T-1)),"cons9")

m.addConstrs((gp.quicksum(Y[l,t] for l in range(L)) == 1 for t in range(T-1)),"cons13")

m.addConstrs((gp.quicksum(X[j,t] for j in range(J)) <=\
              gp.quicksum(Q[l][t]*Y[l,t] for l in range(L)) for t in range(T-1)),"cons10")

#m.addConstrs((Z[j,t,i] == Zs[j,t,i] for j in range(J) for t in range(T-1) for i in range(I)), "cons11")

#m.addConstrs((gp.quicksum(X[j,t] for j in range(J)) <=\
#               gp.quicksum(Q[l]*Y[l,t] for l in range(L)) for t in range(T-1)),"cons10")

#m.addConstrs((E[l,t] <= q[l]*Q[t] for l in range(L) for t in range(T)),"cons11")

#m.addConstrs((S[j,t] == S_opt[j][t] for j in range(J) for t in range(T//4)),"cons13")

# Optimize model
m.optimize()

print('Obj: %g' % m.ObjVal)
end = time.time()
print("runtime: ", end-start)

#k=m.getAttr('x', S)

k=m.getAttr('x', Z)
y_list = np.zeros(shape = (L,T))

for l in range(L):
    for t in range(T):
        y_list[l,t] = Y[l,t].X

boilers = np.zeros(shape=y_list.shape[1])
for i in range(y_list.shape[1]):
    if y_list[0][i] == 1:
        boilers[i] = 1
    elif y_list[1][i] == 1:
        boilers[i] = 2
    else:
        boilers[i] = 3

print("boiler levels: ", boilers) 
print("energy consumption cost: ", gp.quicksum(c[l]*Y[l,t].X for l in range(L) for t in range(T-1)))
#print("unmet: ", gp.quicksum(U[j,t].X for j in range(J) for t in range(T-1)))
"""
for t in range(T-1):
    print ("energy: ", t, gp.quicksum(X[j,t].X for j in range (J)))
"""
#print("unmet by t: ", (U[j,t].X for j in range(J) for t in range(T-1)))
#print("unmet cost: ", gp.quicksum((60*U[j,t].X for j in range(J) for t in range(T-1))))
z_list=np.zeros(shape=(J,T)).reshape(J,T)
for j in range(J):
    for t in range(T):
        for i in range(I):
            if k[(j,t,i)]==1:
                z_list[(j,t)] = i
#print(z_list)

m.write("sdhp.lp")
"""
s2=[]
for j in range(J):
    l=[]
    for t in range(T):
        l.append(k[(j,t)])
    s2.append(l)
    
import pickle

with open('s_1012.pkl', 'wb') as f:
    pickle.dump(s2, f)
x 
k=m.getAttr('x', K)

s=[]
for j in range(J):
    l=[]
    for t in range(T):
        l.append(k[(j,t)])
    s.append(l)
    
k=m.getAttr('x', Y)

y=[]
for j in range(L):
    l=[]
    for t in range(T):
        l.append(k[(j,t)])
    y.append(l)
    
k=m.getAttr('x', Z)

z=[]
for j in range(I):
    for t in range(T):
        for i in range(I):
            k[(j,t,i)]
    
"""
"""  
    results[k] = dict()
    results[k]['obj'] = m.objVal
    results[k]['cpu'] = m.Runtime"""

"""
k=m.getAttr('x', S)

s=[]
for j in range(J):
    l=[]
    for t in range(T//4):
        l.append(k[(j,t)])
    s.append(l)

z=m.getAttr('x', Z)
k=dict()
for j in range(J):
    for t in range(T):
        for i in range(I):
            k[j,t,i]=z[(j,t,i)]
            
    
for v in m.getVars():
    if v.varName=='X':
        print('%s %g' % (v.varName, v.x))"""