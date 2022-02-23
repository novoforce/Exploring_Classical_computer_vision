import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import perspective

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()

def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # display(edged)
    return edged

def detect_contours(edged):
    #cv2.RETR_TREE
    cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    return cnts



if __name__ == '__main__':
    imgp =r'images/page.jpg'
    image= cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    orig = image.copy()
    # display(image)

    #get the edges 
    edged= get_edges(image)

    #find the contours from the edges
    cnts= detect_contours(edged)

######################################################################
#The below filteration should be used if using cv2.RETR_TREE for finding contours, for cv2.RETR_EXTERNAL the below code is not required

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # return polygon points

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
########################################################################

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    display(image)

    #getting warped image
    warped = perspective.four_point_transform(orig, screenCnt.reshape(4, 2))
    display(warped)
