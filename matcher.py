import numpy as np
import cv2

def matcher(I: np.ndarray, Y: np.ndarray) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Match keypoints between two grayscale images using ORB and return matched coordinates.
    
    Params
    ----
    - I: np.ndarray, grayscale image
    - Y: np.ndarray, grayscale image
    
    Returns
    ----
    list of tuples: [ ((x1, y1), (x1', y1')), ... ]
    """
    # Detect keypoints and compute descriptors with ORB
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(I, None)
    kp2, des2 = orb.detectAndCompute(Y, None)

    # Use Brute-Force matcher with Hamming distance (suitable for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (optional but improves RANSAC)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute homography with RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # Select inliers only
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    result = [
        (tuple(kp1[m.queryIdx].pt), tuple(kp2[m.trainIdx].pt))
        for m in inlier_matches
    ]

    return result

def draw_matches(img1, img2, matches, output_path="matches.jpg"):
    """
    Dibuja los matches entre dos imágenes y guarda el resultado
    
    Args:
        img1: Primera imagen (BGR)
        img2: Segunda imagen (BGR)
        matches: Lista de matches como [(pt1, pt2), ...]
        output_path: Ruta para guardar la imagen resultante
    """
    # Convertir los puntos al formato KeyPoint
    kp1 = [cv2.KeyPoint(x=pt1[0], y=pt1[1], size=10) for pt1, _ in matches]
    kp2 = [cv2.KeyPoint(x=pt2[0], y=pt2[1], size=10) for _, pt2 in matches]
    
    # Convertir matches al formato DMatch
    dmatch = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(matches))]
    
    # Dibujar matches
    matched_img = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        dmatch, 
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)  # Color verde para los matches
    )
    
    # Guardar imagen
    cv2.imwrite(output_path, matched_img)
    return matched_img