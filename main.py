import cv2
import numpy as np
from matcher import matcher, draw_matches
from Homo import findHomo, getHomografia, maxmin, warp_image, simple_blend
import os
def main():
    try:
        dir_path = "./image/wall_images/"
        
        raw_images = os.path.join(dir_path, "raw_images")
        
        matches_dir = os.path.join(dir_path, "matches")
        
        images = [cv2.resize(cv2.imread(image_path), (800, 600)) for image_path in sorted(os.listdir(raw_images))] # No aguanta mi pc las imagenes por lo que las tuve que redimensionar
        grayscale = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
        
        matches = []
        for index in range(len(grayscale)-1):
            match = matcher(grayscale[index], grayscale[index+1])
            matches.append(match)
            draw_matches(
                images[index], images[index+1], 
                match,
                os.path.join(matches_dir, f"matches_{index}_{index+1}.jpg")
            )
            
        H_list = findHomo(matches)
        
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
        
        cv2.imwrite(f"{dir_path}output.jpg", panorama)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

main()