import numpy as np
import cv2

def matcher(I: np.ndarray, Y: np.ndarray) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(I, None)
    kp2, des2 = orb.detectAndCompute(Y, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    result = [
        (tuple(kp1[m.queryIdx].pt), tuple(kp2[m.trainIdx].pt))
        for m in inlier_matches
    ]

    return result

def draw_matches(img1, img2, matches, output_path="matches.jpg"):
    
    kp1 = [cv2.KeyPoint(x=pt1[0], y=pt1[1], size=10) for pt1, _ in matches]
    kp2 = [cv2.KeyPoint(x=pt2[0], y=pt2[1], size=10) for _, pt2 in matches]
    
    dmatch = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(matches))]
    
    matched_img = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        dmatch, 
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)  
    )
    
    cv2.imwrite(output_path, matched_img)
    return matched_img