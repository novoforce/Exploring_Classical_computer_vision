import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()


def harris_corners(img):
    gray_flat_chess = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    display(gray_flat_chess)
    gray = np.float32(gray_flat_chess)#convert to float32 as cv2.cornerHarris requires
    dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04) #src, neighborhood_Size, sobel_k_size, harris free params
    ##############################################corner harris finishes
    #################post processing steps
    display(dst)
    dst = cv2.dilate(dst,None) #nms

    # print("len:> ",dst[dst==dst.max()],len(dst[dst==dst.max()]))
    # print("index:> ",np.where(dst== dst.max()))
    # img[dst == dst.max()]=[255,0,0]
    # img[[255,206,158]]= [255,255,0]

    img[dst > 0.10*dst.max()]=[0,0,255] #RGB

    for pt in zip(*np.where(dst == dst.max())): #finding all the index locations
        print(pt)
        x,y = pt
        img[x,y] = [255,0,0]

    # display(dst)
    display(img)




if __name__ == '__main__':
    img1_p= r'images/flat_chessboard.png'
    img2_p= r'images/real_chessboard.jpg'

    flat_chess = cv2.imread(img1_p)
    flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
    
    # display(gray_flat_chess)


    real_chess = cv2.imread(img2_p)
    real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)
    # gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)
    # display(gray_real_chess)

    # harris_corners(flat_chess)
    harris_corners(real_chess)