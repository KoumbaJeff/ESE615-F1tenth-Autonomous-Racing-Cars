import cv2
import numpy as np
import glob

def calculate_distance(intrinsic_matrix, x, y, mounting_height):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    x_car = mounting_height / (y - cy) * fy
    y_car = (x - cx) * (-x_car) / fx
    
    return x_car, y_car

checkerboard = (6, 8)
side_width = 0.25

# Get camera intrinsix matrix
rw_points = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
rw_points[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2) * side_width
img_points = []

images = glob.glob("calibration" + '/*.png')

for file in images:
    img = cv2.imread(file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_img, checkerboard, None)
    
    if ret:
        img_points.append(corners)

    #     img = cv2.drawChessboardCorners(img, checkerboard, corners,ret)
    #     cv2.imshow('Chessboard',img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

ret, intrinsic_matrix, distortion, rotation_vecs, translation_vecs = cv2.calibrateCamera([rw_points] * len(img_points), img_points, gray_img.shape[::-1], None, None)

print("\nCamera matrix : \n", intrinsic_matrix) 

# Get mounting height
# cone_img = cv2.imread("resource/cone_x40cm.png")
# cv2.imshow("Select Cone", cone_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cone_img = cv2.imread("resource/cone_unknown.png")
# cv2.imshow("Select Cone", cone_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread("resource/cone_x40cm.png")
x, y = 665, 500 # Pixel location of cone
x_car = 0.4

fy = intrinsic_matrix[1, 1]
cy = intrinsic_matrix[1, 2]
mounting_height = (y - cy) * x_car / fy
print("Mounting height of camera: ", mounting_height)

img = cv2.imread("resource/cone_unknown.png")
x, y = 600, 415 # Pixel location of cone
x_car, y_car = calculate_distance(intrinsic_matrix, x, y, mounting_height)
print("Estimated distance to cone: ", x_car, y_car)