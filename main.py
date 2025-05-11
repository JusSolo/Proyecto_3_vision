import cv2
import numpy as np
from matcher import matcher
from Homo import findHomo, getHomografia, maxmin, warp_image, simple_blend

def main():
    try:
        images = [cv2.resize(cv2.imread(f"muro{i+1}.jpg"), (800, 600)) for i in range(3)] # No aguanta mi pc las imagenes por lo que las tuve que redimensionar
        grayscale = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
        
        matches12 = matcher(grayscale[0], grayscale[1])
        matches23 = matcher(grayscale[1], grayscale[2])
        H_list = findHomo([matches12, matches23])
        
        Hic = getHomografia(H_list, ic=1)
        xmin, xmax, ymin, ymax = maxmin(images, Hic)
        
        panorama_size = (int(ymax-ymin), int(xmax-xmin), 3)
        print(f"Tamaño panorama: {panorama_size[1]}x{panorama_size[0]}")
        
        warped_images = []
        masks = []
        T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
        
        for i, img in enumerate(images):
            print(f"Procesando imagen {i+1}...")
            H_total = T @ Hic[i]
            warped = warp_image(img, H_total, panorama_size)
            warped_images.append(warped)
            masks.append((warped.max(axis=2) > 0).astype(np.uint8)*255)
        
        panorama = simple_blend(warped_images, masks)
        
        cv2.imwrite("panorama_resultado.jpg", panorama)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

main()