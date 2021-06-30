import cv2 as cv
import mediapipe as mp
import time

# Get webcam
cam = cv.VideoCapture(0)

# Initialize hand tracking
mpHand = mp.solutions.hands
hands = mpHand.Hands()

# Draw
drawHand = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# Read frame
while True:
    success, img = cam.read()

    # To RGB
    rgbImage = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Track
    result = hands.process(rgbImage)

    # print(result.multi_hand_landmarks)

    # Get info about hands and draw
    if result.multi_hand_landmarks:
        for handLM in result.multi_hand_landmarks:
            for id, lm in enumerate(handLM.landmark):
                # print(id, lm)

                # Get position
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)

                # Highlight landmark
                if id==2:
                    cv.circle(img, (cx,cy), 10, (255,0,255), cv.FILLED)

            drawHand.draw_landmarks(img, handLM, mpHand.HAND_CONNECTIONS)


    # Calculate FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Display FPS
    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

    # Show frame
    cv.imshow("Image", img)
    cv.waitKey(1)