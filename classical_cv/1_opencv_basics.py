#read image in opencv format
#plot the image in matplotlib format

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image_cv2(filename):
    img= cv2.imread(filename)

    # plotting the image using opencv format
    while True:
        cv2.imshow('golden_retriever',img)
        # if i waited for 1 sec and i pressed q then break
        if cv2.waitKey(1) & 0xFF == ord('q'): #haxadecimal constant
            break

    cv2.destroyAllWindows()
    return img



def flip_img(image):
    flip_img_h= cv2.flip(image,0) #0:horizinal mirror , 1:vertical mirror

    plt.imshow(flip_img_h)
    plt.show()

    flip_img_v= cv2.flip(image,1) #0:horizinal mirror , 1:vertical mirror , -1:both

    plt.imshow(flip_img_v)
    plt.show()

def resizing_img(image):
    
    print(f"shape of the image:> {image.shape}") #h,w,d
    # plt.imshow(image)
    # plt.show()
    new_img1= cv2.resize(image.copy(),(300,739)) #width,height because this is numpy array
    # plt.imshow(new_img1)
    # plt.show()

    print("the new shape is :> ",new_img1.shape)

    # import sys
    # sys.exit()
    #################################################
    ratio_w= 0.5
    ratio_h= 0.5
    new_img2= cv2.resize(image.copy(),(0,0), image.copy(), ratio_w, ratio_h) #w,h
    # plt.imshow(new_img2)
    # plt.show()
    print(f"shape of the image:> {new_img2.shape}")
    ###################################################
    return new_img2


def read_image(filename):
    img= cv2.imread(filename) #(height,width,depth) #always return a numpy array (BGR)

    #checking condition
    if not isinstance(img,np.ndarray):
        print("pls give correct path")
        return

    # plt.imshow(img)# draw the image RGB
    # plt.show() #render on the screen

    #images would be in the wierd color space
    #matplotlib: RGB, OPENCV: BGR
    img1= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change the color space

    # plt.imshow(img1)# draw the image
    # plt.show() #render on the screen


    #convertion to the grayscale image
    img2= cv2.imread(filename, cv2.IMREAD_GRAYSCALE) #B/W IMAGE DATA

    # plt.imshow(img2)# draw the image
    # plt.show() #render on the screen
    #wierd color
    # plt.imshow(img2,cmap='gray')# cmap="gray", "magma"
    # plt.show() #render on the screen

    print(f"shape of the image:> {img.shape} \ntype of the image:> {type(img)}")

    return img1







if __name__ == '__main__':

    filename = r'D:\Exploring-Tensorflow\classical_cv\images\golden_retriever.jpg'
    # img1= read_image(filename)

    # img2= resizing_img(img1)

    # flip_img(img2)

    read_image_cv2(filename)
