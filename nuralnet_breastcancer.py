import pandas as pd
import numpy as np
import time


dataframe=pd.read_csv('breastcancer.csv')
x_input=dataframe.drop(['id','class1','f6'],axis=1)

x=x_input.loc[:,['f1','f2','f3','f4','f5','f7','f8','f9']].as_matrix()
T_input=(dataframe.drop(['id','f1','f2','f3','f4','f5','f6','f7','f8','f9'],axis=1))
T=T_input.loc[:,['class1']].as_matrix()
n_in=8
n_hidden=8
n_out=1
learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

def sigmoid(x):
       out=(1.0)/(1+np.exp(-x))
       return (out)
def sigmoidprime(x):
       out=(1.0)/(1+np.exp(-x))
       return (out*(1-out))
def train(n,t,V,W,bv,bw):
       A=np.dot(x,V)+bv
       Z=sigmoid(A)
       B=np.dot(Z,W)+bw
       Y=sigmoid(B)

       Ew=(Y-t)
       Ev=sigmoidprime(A)*np.dot(W,Ew)

       dw=np.outer(Z,Ew)
       dv=np.outer(x,Ev)

       loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )

       return loss,dv, dw, Ev, Ew
V = np.random.normal( size=(n_in, n_hidden))
W = np.random.normal( size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)


for epoch in range(250):
    err = []
    upd1 =[0]
    upd2 =[0]
    upd3 =[0]
    upd4 =[0]

    t0 = time.clock()
    for i in range(x.shape[0]):
        loss, a,b,c,d = train(x[i], T[i],V,W,bv,bw )

       
        V-=upd1
        W-=upd2
        bv-=upd3
        bw-=upd4

        upd1 = learning_rate * a+ momentum * upd1
        upd2 = learning_rate * b+ momentum * upd2
        upd3 = learning_rate * c+ momentum * upd3
        upd4 = learning_rate * d+ momentum * upd4

        err.append( loss )

    print ("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))

              
       
       
