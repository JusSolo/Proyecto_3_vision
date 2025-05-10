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
    if ic == -1: # por defecto la imagen central es la de enmedio de la lista
        ic = len(H)//2
    Hic = []
    for i in range(len(H)):
        T = np.eye(3)
        if i < ic:
            T = H[0]
            for j in range(1,i):
               T = T @ H[j]
        if i >= ic:
            if i == ic:
                Hic.append(T.copy())
            for j in range(ic, i):
                T = T @ np.linalg.inv(H[j])
        Hic.append(T)
    return Hic


def maxmin(Lista,ic,Hic):
    xmin, xmax = 0 , 0
    ymin, ymax = 0 , 0
    print(len(Lista) , len(Hic))
    for i in range( len(Lista)):
        L, C, _ = Lista[i].shape
        H = Hic[i]
        esquinas = [(0,0),(0,C-1),(L-1,0),(L-1,C-1)]
        for x, y in esquinas:
            p = np.array([x, y, 1])  # coordenadas homogéneas
            z = H @ p
            z /= z[2]

            if z[0] < xmin: xmin = z[0]
            if z[0] > xmax: xmax = z[0]
            if z[1] < ymin: ymin = z[1]
            if z[1] > ymax: ymax = z[1]

    return xmin, xmax, ymin, ymax


def getfondo(xmin, xmax, ymin, ymax):
    # genesa una imagen negra donde vamos a pegar las fotos
    ancho = int(np.ceil(xmax - xmin))
    alto = int(np.ceil(ymax - ymin))
    fondo = np.zeros((alto, ancho, 3), dtype=np.uint8)  # Imagen RGB negra
    return fondo


def getCoorAfondo(xmin, ymin):
    # Traslación para que el punto (xmin, ymin) quede en (0, 0)
    # devuelve la matriz entre las coordenadas (x,y,1) {las coordenadas que devuelven las homografías} a (x,y,1) posicion del pixcel en el fondo
    T = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ])
    Tinv = np.array([
        [1, 0, xmin],
        [0, 1, ymin],
        [0, 0, 1]
    ])

    return T , Tinv

def color_choice(p_fondo, Tinv, Hicinv, Fotos):
    """
    Devuelve el color (RGB) para un píxel dado en el fondo,
    basado en UNA de las imágenes originales y sus homografías inversas

    p_fondo: np.array([x, y, 1]) -> coordenada en el fondo NOOLVIDAR AGREGAR EL 1 PUES SON COORDENADAS PROYECTIVA
    Tinv: matriz que convierte coordenadas del fondo al sistema común (cootdenadas en la imagen central)
    Hiciv: lista de homografías inversas has np.linalg.inv(Hic[i]) para invertirlas
    Fotos: lista de imágenes originales a color
    """
    # Coordenada en el sistema común (coordenada global)
    p_global = Tinv @ p_fondo
    p_global /= p_global[2]

    for i in range(len(Fotos)):
        p_img = Hinv @ p_global
        p_img /= p_img[2]
        # en esta parte se hace una estrategia de blending###################################
        x_img, y_img = int(round(p_img[0])), int(round(p_img[1]))                           #
                                                                                            #  Esta es la estratrgia
        alto, ancho = Fotos[i].shape[:2]                                                    # mas simple
        if 0 <= x_img < ancho and 0 <= y_img < alto:                                        # no toma en cuenta los colores de
            return Fotos[i][y_img, x_img]  # Nota: filas = y, columnas = x                  # los pixeles vecinos ni, el color de
############################################################################################# las otras fotos si hay traslape
    return np.array([0, 0, 0], dtype=np.uint8)  # negro si ningún pixel válido
