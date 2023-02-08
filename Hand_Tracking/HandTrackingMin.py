import cv2
import mediapipe as mp
import time
from pyfiglet import Figlet


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
prev_text = Figlet(font='slant')
mpDraw = mp.solutions.drawing_utils
pTime = 0 # pTime - previous time
cTime = 0 # pTime - current time



while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    print(id, lm)
                    h,w,c = img.shape
                    cx, cy = int (lm.x*w), int (lm.y*h)
                    print (id, cx, cy)
                    if id == 0: # big finger
                        cv2.circle(img, (cx, cy), 30, (10, 0, 255), cv2.FILLED)
                    if id == 4: # big finger
                        cv2.circle(img, (cx, cy), 7, (20, 0, 255), cv2.FILLED)
                    if id == 8: # big finger
                        cv2.circle(img, (cx, cy), 10, (40, 0, 255), cv2.FILLED)
                    if id == 12: # big finger
                        cv2.circle(img, (cx, cy), 10, (200, 0, 255), cv2.FILLED)
                    if id == 16: # big finger
                        cv2.circle(img, (cx, cy), 15, (155, 0, 255), cv2.FILLED)
                    if id == 20: # big finger
                        cv2.circle(img, (cx, cy), 20, (225, 0, 255), cv2.FILLED)


                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # print('pTime', pTime , 'cTime: ', cTime)

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)

        # print(prev_text.renderText(results.multi_hand_landmarks))


        cv2.imshow("Image", img)

        cv2.waitKey(1)

