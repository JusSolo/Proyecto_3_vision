import numpy as np

def findHomo(L):
    H = []
    for l in L:
        A = np.zeros((len(l)*2,9),float)
        for i in range(0,len(l),2):
            X , X_= l[i]
            x , y  = X
            x_, y_ = X_

            A[i]   =  [0,0,0,x,y,1,-x*y_,-y*y_,-y_]
            A[i+1] =  [x,y,1,0,0,0,-x_*x,-y*x_,-x_]
        U,S,V = np.linalg.svd(A)
        _, _, V = np.linalg.svd(A)
        h = V[-1].reshape(3, 3)
        H.append(h)
    return H

def getHomografia(H,ic = -1):
    if ic == -1:
        ic = len(H)//2
    Hic = []
    for i in range(len(H)):

         T = np.eye(3)

        if i <i:
            T = H[0]
           for j in range(1,i):
               T = T @ H[j]
        if i > ic:

            for j in range(ic, i):
                T = T @ np.linalg.inv(H[j])
        Hic.append(T)
    return Hic
