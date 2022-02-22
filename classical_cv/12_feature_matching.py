"""
Feature Matching steps:
1. Read the 2 images
2. compute the feture map for 2 images
3. use the matcher function to sort the similar features based on distance metric
4. draw the detected/matched features on the image

ORB decriptors:> Oriented FAST and Rotated BRIEF
SIFT descriptors:> Scale-Invariant Feature Transform (patented)
SURF descriptors:> Speeded-Up Robust Features (patented)
"""

from sys import flags
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()


def bf_detection_orb(img1,img2):
    orb = cv2.ORB_create()

    #kp1==keypoint coordinate object, des1== description of the feature
    kp1, des1 = orb.detectAndCompute(img1,None) #image , mask
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2) #match object

    #post-processing
    #sorting the match based on the distance. less distance more match
    matches = sorted(matches, key = lambda x:x.distance)

    final_matches = cv2.drawMatches(img1,kp1,img2,kp2,matches[:25],None) #mask is none
    display(final_matches)


def br_detection_sift(img1,img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2) #top2 matches for each descriptors
    # print("matches:> ",matches[0][0].distance,matches[0][1].distance)

    # post processing
    good = []
    #lowe's ratio test
    for match1,match2 in matches:
        if match1.distance/ match2.distance < 0.50 :
            good.append([match1])
    sift_matches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    #for drawing bounding box

    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good]).reshape(-1,1,2)  
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) #matrix, mask
    # print("---> ",mask)
    # matchesMask = mask.ravel().tolist()

    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)

    dst = cv2.perspectiveTransform(pts,M)
    dst += (w, 0)  # adding offset

    img3 = cv2.polylines(sift_matches, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)

    # print("points after reshape:> ",img1)


    display(img3)


if __name__ == '__main__':
    imgp1= r'images/lucky_charms.jpg'
    img1= cv2.imread(imgp1,0)
    imgp2= r'images/many_cereals.jpg'
    img2= cv2.imread(imgp2,0)

    #Brute force detection using ORB descriptors
    # bf_detection_orb(img1,img2)

    #Brute force detection using SIFT descriptors
    br_detection_sift(img1,img2)

