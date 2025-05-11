import numpy as np
import cv2

def findHomo(L):
    H = []
    for matches in L:
        A = []
        for (x1, y1), (x2, y2) in matches:
            A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
            A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        
        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H_matrix = V[-1].reshape(3, 3)
        H_matrix /= H_matrix[2, 2]  # Normalización
        H.append(H_matrix)
    return H

def getHomografia(H_list, ic=None):
    n = len(H_list) + 1
    if ic is None:
        ic = n // 2
    
    Hic = [np.eye(3) for _ in range(n)]
    
    for i in range(ic-1, -1, -1):
        Hic[i] = H_list[i] @ Hic[i+1]
        
    for i in range(ic+1, n):
        Hic[i] = np.linalg.inv(H_list[i-1]) @ Hic[i-1]
        
    return Hic

def maxmin(images, Hic):
    x_coords, y_coords = [], []
    
    for img, H in zip(images, Hic):
        h, w = img.shape[:2]
        corners = np.array([[0,0,1], [w-1,0,1], [w-1,h-1,1], [0,h-1,1]]).T
        transformed = H @ corners
        transformed /= transformed[2, :]
        x_coords.extend(transformed[0, :])
        y_coords.extend(transformed[1, :])
    
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def warp_image(img, H, output_shape):
    return cv2.warpPerspective(
        img, H, (output_shape[1], output_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

def build_laplacian_pyramid(img, levels=5):
    current = img.astype(np.float32)
    pyramid = [current]
    
    for _ in range(levels-1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid

def simple_blend(images, masks):
    result = np.zeros_like(images[0], dtype=np.float32)
    total_weight = np.zeros(images[0].shape[:2], dtype=np.float32)
    
    for img, mask in zip(images, masks):
        weight = mask.astype(np.float32)/255.0
        result += img.astype(np.float32) * weight[..., np.newaxis]
        total_weight += weight
    
    total_weight[total_weight == 0] = 1  # Evitar división por cero
    return (result / total_weight[..., np.newaxis]).astype(np.uint8)