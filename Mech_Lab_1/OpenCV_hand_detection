import cv2
import numpy as np
import math 

cap = cv2.VideoCapture(0)

#check is camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Press 'q' to exit.")

x0, y0 = 100, 100   # top-left corner
x1, y1 = 300, 300   # bottom-right corner

while True:
    
    ret, frame = cap.read()
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)   #dont hard code boundaries
    crop_frame = frame[y0:y1, x0:x1]
    

    # for exiting the window, since the steam is contasntly opening new windows, then you need to press q to exit or ctrl c the terminal fyi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #canny vs otsus
    #thresh1  = cv2.Canny(blur, threshold1=50, threshold2=255)
    _, thresh1 = cv2.threshold(blur, 127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #test frame
    #cv2.imshow('Live Camera Feed', edges)

    #opencv contour function, made simplified from the video we watched
    contours, hierarchy = cv2.findContours(
        thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cnt = max(contours, key = lambda x: cv2.contourArea(x))

     # show image test
    #cv2.imshow('Live Camera Feed', frame)   # actual feed (comparison)
    #cv2.imshow('Edges in ROI', thresh1)       # contour

    #bounding rectangle around the extracted contour (optional?)
    x, y , w , h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_frame, (x,y), (x+w, y+h), (0,0,255), 0)

    #convex hull for hand
    hull = cv2.convexHull(cnt)

    # physically draw in the contours
    drawing = np.zeros(crop_frame.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0,255,0), 0)
    cv2.drawContours(drawing, [hull], 0, (0,0,255), 0)

    hull = cv2.convexHull(cnt, returnPoints= False)

    #convexity defects
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0 
    cv2.drawContours(thresh1, contours, -1 , (0,255, 0),3 )

    #cosine rule to find the angle for all defects (in between the fingers)
    #with the angle between fingers >90, we should ignore defects (fingers arent gonna be that spread out)
    if defects is not None:

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

            #ignore angles that are larger than 90
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_frame, far, 1, [0,0,255], -1)

                cv2.line(crop_frame,start,end, [0,255,0], 2)

#define actions, set value to however many fingers that are being shown off. Then, after this, we can set actions to values for GPIO PINS on the raspberry pi.
    if count_defects == 1:
        cv2.putText(frame, "2 fingers", (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2 , 2)
        val = 2
    elif count_defects == 2:
        cv2.putText(frame, "3 fingers", (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2 , 2)
        #print("3")
        val = 3
    elif count_defects == 3:
        cv2.putText(frame, "4 fingers", (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2 , 2)
        #print("4")
        val = 4
    elif count_defects == 4:
        cv2.putText(frame, "5 fingers", (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2 , 2)
        #print("5")
        val = 5

    else:
        cv2.putText(frame, "This is 1 finger", (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2 , 2)
        #print("1")
        val = 1
    
    cv2.imshow('Gesture', frame)
    all_img = np.hstack((drawing,crop_frame))
    cv2.imshow("Contours", all_img)

   

# When everything is done, release the capture and destroy all windows, not sure if we need this but it was in the video tutorial
cap.release()
cv2.destroyAllWindows()
