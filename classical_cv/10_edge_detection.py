# credits:> https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny
#methods
# blurring-->find gradient and edges-->apply NMS-->apply thresholding given from API
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()

def canny_edge(image):
    edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
    display(edges)

def canny_edge_blur(image):
    blurred_img = cv2.blur(image,ksize=(5,5))
    edges = cv2.Canny(image=blurred_img, threshold1=44 , threshold2=127)
    display(edges)

def canny_slider():
    def callback(x):
        print(x)

    img = cv2.imread(r'images/sammy_face.jpg', 0) #read image as grayscale


    canny = cv2.Canny(img, 85, 255) 

    cv2.namedWindow('image') # make a window with name 'image'
    cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
    cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

    while(1):
        numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
        cv2.imshow('image', numpy_horizontal_concat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: #escape key
            break
        l = cv2.getTrackbarPos('L', 'image')
        u = cv2.getTrackbarPos('U', 'image')

        canny = cv2.Canny(img, l, u)

    cv2.destroyAllWindows()






if __name__ == '__main__':
    imgp= r'images/sammy_face.jpg'
    img= cv2.imread(imgp)

    # canny_edge(img)

    # canny_edge_blur(img)

    canny_slider()


#utility functions

