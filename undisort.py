import cv2
import numpy as np

#преобразование номера в отдельную картинку

def undisort(img_gray, points):
    HEIGHT = 50
    WIDTH = 255
    points_destination = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], np.int32)
    M, mask = cv2.findHomography(points, points_destination)
    return cv2.warpPerspective(img_gray, M, (WIDTH, HEIGHT))

