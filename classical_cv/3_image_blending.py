import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()


def overlay(imgp1,imgp2):
    img1 = cv2.imread(imgp1)
    img2 = cv2.imread(imgp2)
    img2 =cv2.resize(img2,(300,300)) #deleberately

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    large_img = img1
    small_img = img2

    # x_offset=0
    # y_offset=0

    large_img[0:0+small_img.shape[0], 0:0+small_img.shape[1],:] = small_img
    display(large_img)

def vanilla_blending(imgp1,imgp2):
    img1= cv2.imread(imgp1)
    img2= cv2.imread(imgp2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 =cv2.resize(img1,(1200,1200))
    img2 =cv2.resize(img2,(1200,1200))
    blended = cv2.addWeighted(src1=img1,alpha=0.3,src2=img2,beta=0.7,gamma=0) #opencv api
    display(blended)

def blend_image(imgp1,imgp2):
    """
    blending without thresholding as the logo is a solid color
    """
    img1= cv2.imread(imgp1)
    img2= cv2.imread(imgp2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    print("the shape of img1:> ",img1.shape) #hwd
    print("the shape of img2:> ",img2.shape) #hwd
    img2= cv2.resize(img2,(0,0), img2, 0.3, 0.3)

    display(img1)
    display(img2)
    print("the shape of img1:> ",img1.shape) #hwd
    print("the shape of img2:> ",img2.shape) #hwd


    #create a ROI of the img1
    y_offset= img1.shape[0] - img2.shape[0] #height - height
    x_offset= img1.shape[1] - img2.shape[1] #width - width

    roi= img1[y_offset: , x_offset:,:]
    print(f"shape of roi:> {roi.shape}")
    display(roi)


    #########################################################
    # creating a mask
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    print(f"shape of img2gray:> {img2gray.shape}")
    display(img2gray)


    #inverse the mask
    mask_inv = cv2.bitwise_not(img2gray) #only 2 channels
    print(f"after inverting")
    display(mask_inv)
    
    #place the mask on top of img2-- so as to extract the logo part
    fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
    print("after bitwising")
    display(fg)

    #final roi image
    final_roi= cv2.add(roi,fg) #cv2.bitwise_or() will work too
    display(final_roi)

    # import sys
    # sys.exit()

    img1[y_offset: , x_offset:,:]= final_roi

    display(img1)


def blend_image2(imgp1,imgp2):
    """
    blend the 2 images using thresholding
    thresholding is helpful multi-color logo
    """
    img1 = cv2.cvtColor(cv2.imread(imgp1),cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(imgp2),cv2.COLOR_BGR2RGB)
    img2= cv2.resize(img2,(0,0), img2, 0.3, 0.3)
    display(img1)
    display(img2)
    #creating the ROI of the img1
    y_offset= img1.shape[0] - img2.shape[0] #height - height
    x_offset= img1.shape[1] - img2.shape[1] 
    roi= img1[y_offset:, x_offset:]
    display(roi)

    #gray scaling and thresholding the img2
    img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    display(img2gray)
    ret, mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)
    display(mask)

    # import sys
    # sys.exit()

    #masking out the fg from the img2
    fg = cv2.bitwise_or(img2,img2, mask = mask)
    display(fg)

    roi = cv2.bitwise_or(roi,fg)
    display(roi)

    #place back roi to the original image
    img1[y_offset:, x_offset:] = roi

    display(img1)

def blend_image3(imgp1,imgp2):
    """
    merge the fg of the img2 + bg of img1
    """
    img1 = cv2.cvtColor(cv2.imread(imgp1),cv2.COLOR_BGR2RGB) #original image
    img2 = cv2.cvtColor(cv2.imread(imgp2),cv2.COLOR_BGR2RGB) #logo
    img2= cv2.resize(img2,(0,0), img2, 0.3, 0.3)

    #creating the ROI of the img1
    y_offset= img1.shape[0] - img2.shape[0] #height - height
    x_offset= img1.shape[1] - img2.shape[1] 
    roi= img1[y_offset:, x_offset:]
    display(roi)
    #create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY) #logo to b&w

    ret, mask = cv2.threshold(img2gray, 225, 255, cv2.THRESH_BINARY) #fg
    kernel = np.ones((3,3),np.uint8) #kernel is white, so erode white from borders 
    mask = cv2.erode(mask,kernel,iterations = 1)
    display(mask)

    # get the background of the img1(ROI)
    img1_bg = cv2.bitwise_or(roi,roi,mask = mask)
    display(img1_bg)

    mask_inv = cv2.bitwise_not(mask) #bg
    display(mask_inv)

    #get the foreground of the img2
    img2_fg = cv2.bitwise_or(img2,img2,mask = mask_inv)
    display(img2_fg)

    #merge the fg of the img2 + bg of img1
    final_roi = cv2.add(img1_bg,img2_fg)
    display(final_roi)

    # replace final_roi to the img1
    img1[y_offset:, x_offset:] = final_roi

    display(img1)



if __name__ == '__main__':
    img1= r'images\golden_retriever.jpg'
    img2= r'images\watermark_no_copy.png'
    img3= r'images\cropped-hello.png'
    # vanilla_blending(img1,img2)

    # overlay(img1,img2)

    # blend_image(img1, img2)
    # blend_image2(img1,img3)
    blend_image3(img1,img3)
