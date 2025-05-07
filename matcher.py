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
