import cv2
import numpy as np
import matplotlib.pyplot as plt
#thresholding is only applied on BW images as we need only the features.
#a very simple method to separate features of the image based on the intensity values

#a good resource to checkout: https://docs.opencv.org/3.4/db/d8e/tutorial_threshold.html

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp, vmin=0, vmax=255)
    plt.show()

def binary_threshold(image):
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #threshold, #max_value
    display(thresh1)
    cv2.imwrite('./binary_threshold.png',thresh1)


def binary_threshold_inverse(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    display(thresh2)
    cv2.imwrite('./binary_threshold_inv.png',thresh2)

def binary_threshold_truncate(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC) #if gretaer than threshold: 127(threshold) else ignore max_value
    display(thresh2)
    cv2.imwrite('./binary_threshold_trunc.png',thresh2)

def binary_threshold_to_zero(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO) #if less than threshold: 0 else ignore max_value
    display(thresh2)
    cv2.imwrite('./binary_threshold_to_zero.png',thresh2)

def binary_threshold_to_zero_inverse(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    display(thresh2)
    cv2.imwrite('./binary_threshold_to_zero_inv.png',thresh2)


def adaptive_threshold(image):
    img = cv2.imread(image,0)
    # display(img)

    #simple thresholding
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    display(th1)


    #adaptive thresholding
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
    # display(th2)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,8)
    # display(th3)


    blended = cv2.addWeighted(src1=th2,alpha=0.7,src2=th3,beta=0.3,gamma=0)
    # blended= cv2.bitwise_and(th2,th3)
    display(blended)

















if __name__ == '__main__':
    img_filename= r'images/rainbow.jpg'
    # img = cv2.imread(img_filename,0) #read the image in B/W format
    # display(img)
    # while True:
    #     cv2.imshow('golden_retriever',img)
    #     #if i waited for 1 sec and i pressed q then break
    #     if cv2.waitKey(1) & 0xFF == ord('a'): #haxadecimal constant
    #         break

    # cv2.destroyAllWindows()
    # binary_threshold(img)

    # binary_threshold_inverse(img)

    # binary_threshold_truncate(img)

    # binary_threshold_to_zero(img)

    # binary_threshold_to_zero_inverse(img)

    ##############################################################################333
    img_file= r"D:\Exploring-Tensorflow\classical_cv\images\crossword.jpg"
    adaptive_threshold(img_file)


