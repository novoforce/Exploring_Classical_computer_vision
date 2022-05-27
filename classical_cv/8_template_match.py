# mathematical functions for TM: https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()

def template_match(image,template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape[::] 
    display(img_gray)
    display(template_gray)

    method= ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    res = cv2.matchTemplate(img_gray,template_gray,cv2.TM_CCOEFF) #similarity scores
    # print("----->",res.shape,img_gray.shape,template_gray.shape) 
    display(res)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc  #Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, 255, 2)  #White rectangle with thickness 2. 
    display(image)


def template_match_multi(image,template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = template.shape[::]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # display(res)
    # print("***>",res.shape,len(res),type(res),res)
    threshold = 0.99#heuristic
    loc = np.where(res >= threshold) #filter
    # print("***>",type(loc),loc,*loc)

    for pt in zip(*loc): 
        # print(pt)
        cv2.rectangle(image, pt[::-1], (pt[1] + w, pt[0] + h), (0,0, 255), 1)
    display(image)






if __name__ == '__main__':
    # full = cv2.imread(r'D:\Exploring_Classical_computer_vision\classical_cv\images\sammy.jpg')
    # full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
    # face= cv2.imread(r'D:\Exploring_Classical_computer_vision\classical_cv\images\sammy_face.jpg')
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # template_match(full,face)

    #another example
    img_rgb = cv2.cvtColor(cv2.imread(f'D:\Exploring_Classical_computer_vision\classical_cv\images\hearts.png'), cv2.COLOR_BGR2RGB)
    template = cv2.imread(r'D:\Exploring_Classical_computer_vision\classical_cv\images\hearts_template.png',0)

    template_match_multi(img_rgb,template)