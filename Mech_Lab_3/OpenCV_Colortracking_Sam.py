import cv2
import numpy as np
import time 
import serial

cap = cv2.VideoCapture(0)

ser = serial.Serial('/dev/ttyACM0',9600)
baud_rate = 9600

if not cap.isOpened():                      #error loop
    print("Error: Could not open camera.")
    exit()

print("Camera is Open")

#------preset variables-----
prev_frame_time = 0
new_frame_time = 0
prev_e = 1000
ie = 0

while True:
    ret, frame = cap.read()   #FRAME VARIABLE
    if not ret:
        print("Error: Could not read frame.")
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    del_t = fps /fps
    #print(f"FPS: {fps}")

    #----Convert the Image to the HSV Color Space----

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # HUE (0-179 color), Saturation (0-255 Saturation color strength), Value (0-255 brightness)
    #Hue Range: Red = 0-10 and 170-180. Orange 10-25. Yellow 35-85. Blue 100-130. Purple 130-160.
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])


    # Note: For colors that wrap around the hue spectrum (like red),
    # you may need two separate masks and combine them with a bitwise OR operation.

    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    #cv2.imshow("HSV Camrea Feed", mask)  #----------------Testing HSV filtering 

    #------object deteciton protion of coding-------

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)           #filtering the image of noise 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #image, mode, method (image,The contour retrieval mode determines the hierarchy of the retrieved contours, The contour approximation method This determines how the contour points are stored. )
    #returns contours: which is a nested list of x,y coords for each contour, and hierarchy which relays info about their relationships.
   

    if len(contours) > 0: #checking if there are any contours to begin with
        cnt = max( contours, key = cv2.contourArea)  #iterate over all contours, key being the comparison method.


        #compute the centroid 
        M = cv2.moments(cnt)  #The result is a dictionary M containing keys like "m00", "m10", "m01", and others. m00 represents the 0th order moment which is the area of the contour, m10 and m01 are first order moments which are the x and the y positions.
        if M["m00"] != 0:   #if the largest contoured area is zero then we skip this frame
            x_pos = int(M["m10"] / M["m00"])
            y_pos = int(M["m01"] / M["m00"])       #centroid calculation (calc 2 refresher)

            cv2.circle(frame, [x_pos,y_pos], 7, (0, 255, 0), -1)

            # print(x_pos)   #range of 0 to about 630?   get the distance from the edges of the camera
            # left_dist = 630 - x_pos     #from 630
            # right_dist = x_pos        #from 0
            # frame_width = left_dist + right_dist

            # if x_pos > 280 and x_pos < 340:
            #     print(f"In the middle! Good! {x_pos}")
            
            # elif x_pos < 280 :
            #     print(f"Too far right, turn left {x_pos}")

            # elif x_pos > 340:
            #     print(f"Too far left, turn right {x_pos}")



        #--------proportional error--------    
        if x_pos < 280:
            e = x_pos - 280         #right is negative error
        
        elif x_pos > 280 and x_pos < 340:
            e = 0
        
        elif x_pos > 340:
            e = x_pos - 340         #left is positive error

        
        #-------derivative error----------
        if prev_e == 1000:
            prev_e = e
            de = (e - prev_e) / (del_t)  #the error over the amount of time between frames
        
        else:
            de = (e - prev_e) / (del_t)  #the error over the amount of time between frames
            prev_e = e 

        #------integral error-------------
        ie += ((prev_e + e)/2) * (del_t)


        #-------PID Constants------------
        Kp = 1
        Ki = 0
        Kd = 1

        u = Kp*e + Kd*de +Ki*ie

        print(u)
        u += 290
        byte = int( u / 2.27)  #get info to one byte (loss of information)
        byte = byte.to_bytes(1, byteorder='big', signed=False)
        ser.write(byte)

    cv2.imshow("Ball Locator", frame)

    




cap.release()
cv2.destroyAllWindows()



