import numpy as np
import matplotlib.pyplot as plt
import cv2
def display(image):
    plt.imshow(image)
    plt.show()

def draw_rectangle(image):
    cv2.rectangle(image,pt1=(200,200),pt2=(300,300),thickness=2,color=(255,0,0)) #RGB
    cv2.rectangle(image,pt1=(400,10),pt2=(500,150),thickness=4,color=(255,255,0))
    display(image)
    return image

def draw_circle(image):
    cv2.circle(image,center=(250,250),radius=50,color=(255,255,255),thickness=2)
    cv2.circle(image,center=(400,400),radius=50,color=(50,5,60),thickness=-1)
    display(image)
    return image



def draw_line(image):
    cv2.line(image,pt1=(0,0),pt2=(512,512),color=(255,155,155),thickness=2)
    cv2.line(image,pt1=(512,0),pt2=(0,512),color=(155,255,155),thickness=2)
    display(image)
    return image

def draw_text(image):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image,text="Hello Mahalakshmi uidhiu",
    org=(50,500),fontFace=font,fontScale=1,color=(255,0,0),
    thickness=4,lineType=cv2.LINE_AA)

    display(image)
    return image


def draw_polygon(image):
    vertices= np.array([ [100,300], [200,200], [400,300], [200,400]], dtype=np.int32)
    print(f"shape of the vertices is {vertices.shape}")

    pts= vertices.reshape((-1,1,2)) #
    print(f"shape of the points is {pts.shape}")

    cv2.polylines(image,[pts],isClosed=False, color=(155,0,0),thickness=2)
    plt.imshow(image)
    plt.show()

    return image



#starting point of any python program
if __name__ == '__main__':

    blank_img= np.zeros(shape=(512,512,3), dtype=np.int16)
    # display(blank_img)
    # img= draw_rectangle(blank_img)

    # img2= draw_circle(img)

    # img3= draw_line(img2)

    # img4= draw_text(img3)

    img5= draw_polygon(blank_img)



# assignment
# try drawing a triangle and fill it with some color
