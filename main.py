import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Camera Setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood):

        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = (0, 0)

        # Load Food Image
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)

        if self.imgFood is None:
            raise FileNotFoundError(
                f"Food image '{pathFood}' not found."
            )

        # Resize Food
        self.imgFood = cv2.resize(self.imgFood, (60, 60))

        self.hFood, self.wFood = self.imgFood.shape[:2]

        self.foodPoint = (0, 0)
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = (
            random.randint(100, 1100),
            random.randint(100, 600)
        )

    def update(self, imgMain, currentHead):

        if self.gameOver:

            cvzone.putTextRect(
                imgMain,
                "Game Over",
                [800, 700],
                scale=10,
                thickness=10,
                offset=10
            )

            cvzone.putTextRect(
                imgMain,
                f"Score: {self.score}",
                [500, 400],
                scale=9,
                thickness=9,
                offset=10
            )

            return imgMain

        px, py = self.previousHead
        cx, cy = currentHead

        # Add Head Point
        self.points.append([cx, cy])

        distance = math.hypot(cx - px, cy - py)

        self.lengths.append(distance)
        self.currentLength += distance

        self.previousHead = (cx, cy)

        # Length Reduction
        if self.currentLength > self.allowedLength:

            for i, length in enumerate(self.lengths):

                self.currentLength -= length

                self.lengths.pop(i)
                self.points.pop(i)

                if self.currentLength < self.allowedLength:
                    break

        # Food Collision
        rx, ry = self.foodPoint

        if (
            rx - self.wFood // 2 < cx < rx + self.wFood // 2
            and
            ry - self.hFood // 2 < cy < ry + self.hFood // 2
        ):
            self.randomFoodLocation()

            self.allowedLength += 50
            self.score += 1

            print("Score:", self.score)

        # Draw Snake
        if len(self.points) > 1:

            for i in range(1, len(self.points)):

                cv2.line(
                    imgMain,
                    tuple(self.points[i - 1]),
                    tuple(self.points[i]),
                    (0, 0, 255),
                    20
                )

        if self.points:
            cv2.circle(
                imgMain,
                tuple(self.points[-1]),
                20,
                (0, 255, 0),
                cv2.FILLED
            )

        # Draw Food
        imgMain = cvzone.overlayPNG(
            imgMain,
            self.imgFood,
            (
                rx - self.wFood // 2,
                ry - self.hFood // 2
            )
        )

        # Score
        cvzone.putTextRect(
            imgMain,
            f"Score: {self.score}",
            [50, 80],
            scale=2,
            thickness=2,
            offset=10
        )

        # Self Collision
        if len(self.points) > 20:

            pts = np.array(
                self.points[:-10],
                np.int32
            )

            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(
                imgMain,
                [pts],
                False,
                (0, 255, 0),
                3
            )

            minDist = cv2.pointPolygonTest(
                pts,
                (int(cx), int(cy)),
                True
            )

            if -1 <= minDist <= 1:

                print("Game Over")

                self.gameOver = True

        return imgMain


# Use your image file
game = SnakeGameClass("donut.png")

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(
        img,
        flipType=False
    )

    if hands:

        lmList = hands[0]["lmList"]

        pointIndex = lmList[8][0:2]

        img = game.update(
            img,
            pointIndex
        )

    cv2.imshow("Hand snake", img)

    key = cv2.waitKey(1)

    # Restart Game
    if key == ord("r"):
        game = SnakeGameClass("donut.png")

    # Quit Game
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 