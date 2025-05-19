import cv2
import numpy as np

img = cv2.imread("resource/lane.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([20, 64, 150])
upper_yellow = np.array([30, 255, 255])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 3)

cv2.imshow("Lane Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

