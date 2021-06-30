import cv2 as cv
import mediapipe as mp
import time


class handDetector():

    # Constructor
    def __init__(self, mode=False, maxHands=2, detConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detConf = detConf
        self.trackConf = trackConf

        # Initialize hand tracking
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, self.detConf, self.trackConf)

        # Draw
        self.drawHand = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # To RGB
        rgbImage = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Track
        self.result = self.hands.process(rgbImage)

        # print(result.multi_hand_landmarks)

        # Get info about hands and draw
        if self.result.multi_hand_landmarks:
            for handLM in self.result.multi_hand_landmarks:
                if draw:
                    self.drawHand.draw_landmarks(img, handLM, self.mpHand.HAND_CONNECTIONS)
        return img

    def handPosition(self, img, handNo=0, draw=True):

        listLM = []

        if self.result.multi_hand_landmarks:
            xhand = self.result.multi_hand_landmarks[handNo]

            for id, lm in enumerate(xhand.landmark):

                # Get position
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                listLM.append((id, cx,cy))

                # Highlight landmark
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0 , 0), cv.FILLED)

        return listLM


def main():

    # Get webcam
    cam = cv.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = handDetector()

    while True:
        success, img = cam.read()

        detector.findHands(img)
        lmList = detector.handPosition(img)

        if len(lmList) != 0:
            print(lmList[0])

        # Calculate FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # Display FPS
        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

        # Show frame
        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
