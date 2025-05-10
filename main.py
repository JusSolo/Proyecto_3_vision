import cv2
import numpy as np
from matcher import matcher
from Homo import findHomo, getHomografia, maxmin

def load_and_convert(image_path):
    """Load image and convert to grayscale"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def main():
    # Load images
    try:
        img1 = load_and_convert("muro1.jpg")
        img2 = load_and_convert("muro2.jpg")
        img3 = load_and_convert("muro3.jpg")
    except ValueError as e:
        print(e)
        return

    # Match keypoints between consecutive images
    print("Matching keypoints between images...")
    matches12 = matcher(img1, img2)
    matches23 = matcher(img2, img3)

    print(f"Found {len(matches12)} matches between muro1 and muro2")
    print(f"Found {len(matches23)} matches between muro2 and muro3")

    # Compute homographies
    print("\nComputing homographies...")
    H_list = findHomo([matches12, matches23])
    print("Homography between muro1 and muro2:")
    print(H_list[0])
    print("\nHomography between muro2 and muro3:")
    print(H_list[1])

    # Compute composite homographies
    print("\nComputing composite homographies...")
    Hic = getHomografia(H_list)
    print("Composite homographies:")
    for i, h in enumerate(Hic):
        print(f"Image {i+1} to reference:")
        print(h)

    # Calculate panorama bounds
    print("\nCalculating panorama bounds...")
    images = [cv2.imread(f"muro{i+1}.jpg") for i in range(3)]
    xmin, xmax, ymin, ymax = maxmin(images, 1, Hic)
    print(f"Panorama bounds: x({xmin:.1f}, {xmax:.1f}), y({ymin:.1f}, {ymax:.1f})")

if __name__ == "__main__":
    main()
