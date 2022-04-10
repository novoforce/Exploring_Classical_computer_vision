from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import cv2
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()

def find_edges(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return edged

def detect_paper_contours(edged):
    cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #filters the contours points and get only main points
    for c in cnts:
        # x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) #get the polygon points

        if len(approx) == 4:
            docCnt = approx
            break

    return docCnt

def detect_mcq_contours(warped):
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #filter the questions
    questionCnts = []
    for c in cnts:
        if cv2.contourArea(c) > 850.0:
            (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(warped,(x,y),(x+w,y+h),(0,255,0),2)
            questionCnts.append(c)
    # display(warped)
    return questionCnts,thresh

def find_correct_answers(questionCnts,thresh):
    answers= {}
    kernel = np.ones((5,5),np.uint8)
    for q,i in enumerate(range(0,len(questionCnts),5)):
        #get x sorted contours 
        cont_rows= contours.sort_contours(questionCnts[i:i+5])[0]

        for (j,c) in enumerate(cont_rows):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            # display(mask)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            total = cv2.countNonZero(mask)
            # print('--> ',total)
            #correct answer
            if total > 850:
                answers[q] = (j,c)
    return answers

def check_answers(answers,ANSWER_KEY):
    correct_ans_count= 0
    for (key,value),(ans_k,ans_v) in zip(answers.items(), ANSWER_KEY.items()):
        
        if value[0] == ans_v:
            cv2.drawContours(paper, [value[1]], -1, (0,255,0), 2)
            correct_ans_count+=5
        else:
            cv2.drawContours(paper, [value[1]], -1, (255,0,0), 2)
    return correct_ans_count






if __name__ == '__main__': 
    imgp= r'images/test_05.png'
    image = cv2.cvtColor(cv2.imread(imgp),cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

    #find edges
    edged= find_edges(image)
    display(edged) #check

    #detect paper contours
    docCnt= detect_paper_contours(edged)
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    display(paper) #check

    #get all the MCQ contours
    questionCnts,thresh= detect_mcq_contours(warped)
    display(thresh) #check
    #sorting the contours top-bottom
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    assert(len(questionCnts)==25)
    print("total:> ",len(questionCnts))

    #getting correct answers
    answers= find_correct_answers(questionCnts,thresh)

    #calculate scores
    score= check_answers(answers,ANSWER_KEY)
    print(f"score:>{score}/25")

    #display the score on the paper
    final_score= f"score: {score}/25"
    #create a black patch
    paper[:10+40,:20+200] = 0
    cv2.putText(paper, final_score, (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    display(paper)












