import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()


def chess_corners(image):
    found, corners = cv2.findChessboardCorners(image,(7,7))
    if found:
        cv2.drawChessboardCorners(image, (7, 7), corners, found)
        display(image)


def dot_grid_corners(image):
    found, corners = cv2.findCirclesGrid(image, (10,10), cv2.CALIB_CB_CLUSTERING)
    if found:
        cv2.drawChessboardCorners(image, (10, 10), corners, found)
        display(image)



if __name__ == '__main__':
    imgp= r'images/flat_chessboard.png'
    img= cv2.imread(imgp)
    chess_corners(img)

    imgp= r'images/dot_grid.png'
    img= cv2.imread(imgp)
    dot_grid_corners(img)