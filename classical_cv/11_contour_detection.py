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

def detect_contours_area_sort(image):

    #convert image to gray Scale
    image_g= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #canny edges
    edged= cv2.Canny(image_g,50,200)
    # display(edged)

    #detect contours from edges
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("total contours:> ", len(contours))

    #sorting the contours based on area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True) #descending order

    #plotting the contours
    # RED,GREEN,BLUE,BLACK
    color_dict= {'0':(255,0,0), '1':(0,255,0), '2':(0,0,255), '3':(0,0,0)}
    for i,c in enumerate(sorted_contours):
        cv2.drawContours(image, [c], -1, color_dict[str(i)], 6)
    display(image)


def detect_contours_coord_sort(image):
    #convert image to gray Scale
    image_g= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #canny edges
    edged= cv2.Canny(image_g,50,200)
    # display(edged)
    #detect contours from edges
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("total contours:> ", len(contours))

    #computing the center of mass of each contours
    for c in contours:
        #calculate the moments
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image,(cx,cy), 10, (0,0,0), -1)
    # display(image)

    #sorting function
    def x_cord_contour(contours):
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))

    def y_cord_contour(contours):
        M = cv2.moments(contours)
        return (int(M['m01']/M['m00']))

    sorted_contours = sorted(contours, key=x_cord_contour, reverse=True)#descending

    color_dict= {'0':(255,0,0), '1':(0,255,0), '2':(0,0,255), '3':(0,0,0)}
    for i,c in enumerate(sorted_contours):
        cv2.drawContours(image, [c], -1, color_dict[str(i)], 6)
        #draw ranks on the middle of the contours
        x= x_cord_contour(c)
        y= y_cord_contour(c)
        cv2.putText(image, str(i+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        #crop the contour shapes from the original image
        # (x, y, w, h) = cv2.boundingRect(c)  
        # cropped_contour = image[y:y + h, x:x + w]
        # image_name = "output_shape_number_" + str(i+1) + ".jpg"
        # cv2.imwrite(image_name, cropped_contour)

    display(image)










if __name__ == '__main__':
    imgp= r'images/internal_external.png'
    img = cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    # display(img)
    detect_contours1(img)

    imgp= r'images/hearts.png'
    img = cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    # display(img)
    detect_contours2(img)

    imgp = r'images/bunchofshapes.jpg'
    img = cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    # display(img)
    detect_contours_area_sort(img)

    imgp = r'images/bunchofshapes.jpg'
    img = cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    # display(img)
    detect_contours_coord_sort(img)


