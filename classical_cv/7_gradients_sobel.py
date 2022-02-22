#https://en.wikipedia.org/wiki/Sobel_operator
#https://en.wikipedia.org/wiki/Image_gradient


import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp, vmin=0, vmax=255)
    plt.show()

def sobel_x(img):
    """change of y wrt to x """
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #img,depth,x=1,y=0,kernel(odd)
    display(sobelx)
    return sobelx



def sobel_y(img):
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)#img,depth,x=0,y=1,kernel
    display(sobely)
    return sobely



def laplacian(img):
    """
    laplacian is basically the partial derivative based on x and y values
    so in nutshell it will give net effect of sobel_x + sobel_y(context)
    """
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    display(laplacian)
    return laplacian

def sobelx_plus_sobely(img):
    sobelx= sobel_x(img)
    sobely= sobel_y(img)
    lap= laplacian(img)

    # blended= cv2.addWeighted(sobelx,0.5,sobely,0.5,0) #blending 
    # display(blended)

    # #we can add a threshold to the blended ones
    # _,th= cv2.threshold(blended,210,255,cv2.THRESH_BINARY) #threholding
    # display(th)

    # #gradient thresholding
    # kernel = np.ones((3,3),np.uint8)
    # gradient = cv2.morphologyEx(th,cv2.MORPH_GRADIENT,kernel)
    # display(gradient)

    kernel = np.ones((5,5),np.uint8) #kernel is white, so erode white from borders 
    erosion1 = cv2.dilate(lap,kernel,iterations = 1)
    display(erosion1)



if __name__ == '__main__':
    img_path= r'D:\Exploring-Tensorflow\classical_cv\images\sudoku.jpg'
    img= cv2.imread(img_path,0) #grayscale image
    display(img)
    # sobel_x(img)
    # sobel_y(img)
    # laplacian(img)

    #combining the sobelx and sobely
    sobelx_plus_sobely(img)