

import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

def method():

    x=np.array([1,2,3,4,5,6,7,8])
    y=np.array([4,5,3,7,9,3,4,10])

    interr=squarederror(x,y,0,0)
    b,m=gradientdecsentmethod(x,y,0.001,1,1)
    finerr=squarederror(x,y,b,m)

    plt.scatter(x,y)
    m1=(m[len(m)-1])
    b1=(b[len(m)-1])
    print(m1,b1)
    print(type(m))

    plt.plot(x,m1*x+b1)
    predict=m1*9+b1
    print(predict)




    plt.show()





def squarederror(x,y,b,m):
    sqerror=0
    for i in range(0,len(x)):
        xs=x[i]
        ys=y[i]


        sqerror+=((m*xs+b)-ys)**2

    return(sqerror/2*len(x))
def iterations(x,y,itr,intial_b,intial_m,learningrate):
    b=intial_b
    m=intial_m
    for i in range(0,itr):
      [b,m]=gradientdecsentmethod(x,y,learningrate,b,m)

    return b,m


def gradientdecsentmethod(x,y,learningrate,b,m):
    m_derivative=0
    b_derivative=0
    for i in range(0,len(x)):
        xs=x[i]
        ys=y[i]

        m_derivative+=(1/2)*(y-(m*x+b))*x
        b_derivative+=(1/2)*(y-(m*x+b))

    new_m=m-(learningrate*m_derivative)
    new_b=b-(learningrate*b_derivative)

    return [new_m,new_b]


if __name__ == '__main__':
    method()
