import cv2
import numpy as np
import matplotlib.pyplot as plt
#refer link: https://answers.opencv.org/question/100746/why-in-open-cv-morphology-operations-is-inverted/
#https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm
def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp, vmin=0, vmax=255)
    plt.show()

def create_img():
    blank_img =np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img,text='Deep',org=(50,300), fontFace=font,fontScale= 5,color=(255,255,255),thickness=15,lineType=cv2.LINE_AA)
    return blank_img


def erosion(img):
    #dst(x,y)=min(x′,y′)
    kernel = np.ones((5,5),np.uint8) #kernel is white, so erode white from borders 
    erosion1 = cv2.erode(img,kernel,iterations = 5)
    display(erosion1)

def dilation(img):
    # dst(x,y)=max(x′,y′)
    kernel = np.ones((5,5),np.uint8) #kernel is white, so expand white from borders 
    erosion1 = cv2.dilate(img,kernel,iterations = 2)
    display(erosion1)

def opening(img): #erosion(remove white) + dilation(add white)
    kernel = np.ones((5,5),np.uint8)
    white_noise = (np.random.randint(low=0,high=2,size=(600,600)))* 255
    # display(white_noise)
    noise_img = white_noise+img
    display(noise_img)
    opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
    display(opening)

def closing(img): #dilation(add white) + erosion(remove white)
    kernel = np.ones((5,5),np.uint8)
    black_noise = (np.random.randint(low=0,high=2,size=(600,600)))* -255
    noise_img = black_noise+img
    display(noise_img)
    noise_img[noise_img==-255] = 0 #mathematically stable
    display(noise_img)
    closing = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
    display(closing)



if __name__ == '__main__':
    img= create_img()
    # img= cv2.bitwise_not(img)
    # display(img) #actual image
    # erosion(img) #eroded image
    # dilation(img) #dilate image
    # opening(img)
    closing(img)