#Retrieval modes in findContours:>
# best ref. link on hierarchy:> https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
# hierarchy = [Next,Previous,First Child, Parent]
# * cv2.RETR_EXTERNAL:Only extracts external contours
# * cv2.RETR_CCOMP: Extracts both internal and external contours organized in a two-level hierarchy
# * cv2.RETR_TREE: Extracts both internal and external contours organized in a  tree graph
# * cv2.RETR_LIST: Extracts all contours without any internal/external relationship

#approximations:
# *cv2.CHAIN_APPROX_NONE: returns all the contour points
# *CV2.CHAIN_APPROX_SIMPLE: returns only the minimum contour points


import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()

def detect_contours1(image):
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    display(image_g)
    contours, hierarchy = cv2.findContours(image_g, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("--> ",hierarchy)

    external_contours = np.zeros(image_g.shape)
    for i in range(len(contours)):
        # last column in the array is -1 if an external contour (no contours inside of it)
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(external_contours, contours, i, 255, -1)
    display(external_contours)

    # Create empty array to hold internal contours
    internal_contours = np.zeros(image.shape)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == 0:
            cv2.drawContours(internal_contours, contours, i, (0,0,255), -1)

        if hierarchy[0][i][3] == 4:
            cv2.drawContours(internal_contours, contours, i, (0,255,255), -1) 

    display(internal_contours)

def detect_contours2(image):
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(image_g, 50, 200)
    display(image_g)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)

    #draw contours
    cv2.drawContours(image, contours,-1,(0,0,255),4)
    display(image)






if __name__ == '__main__':
    imgp= r'images/internal_external.png'
    img = cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    # display(img)
    detect_contours1(img)

    imgp= r'images/hearts.png'
    img = cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    display(img)
    detect_contours2(img)

