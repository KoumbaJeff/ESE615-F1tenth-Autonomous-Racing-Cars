import cv2
import numpy as np
import time
from detection import get_engine, label_to_box_xyxy, voting_suppression, bbox_convert_r
import common

# === Camera Intrinsics and Mounting Info ===
intrinsic_matrix = np.array([
    [694.15, 0, 447.88],
    [0, 695.31, 258.29],
    [0, 0, 1]
])
mounting_height = 0.139  # in meters

def calculate_distance(x, y, intrinsic_matrix, mounting_height):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x_car = mounting_height / (y - cy) * fy
    y_car = (x - cx) * (-x_car) / fx
    return x_car, y_car

def draw_bbox(img, bboxs):
    image = img.copy()
    for bbox in bboxs:
        x, y, w, h = bbox
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)  # Red box
    return image

def detect_lanes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 64, 150])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lane_image = image.copy()
    cv2.drawContours(lane_image, contours, -1, (0, 255, 0), 2)
    return lane_image

# === TRT Engine Setup ===
onnx_file_path = "f1tenth_model.onnx"
engine_file_path = "f1tenth_model.trt"
input_shape = (180, 320)
engine = get_engine(onnx_file_path, engine_file_path)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = common.allocate_buffers(engine)

# === Camera Setup ===
cam = cv2.VideoCapture("/dev/video4")
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cam.set(cv2.CAP_PROP_FPS, 60)

# === Main Loop ===
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = frame.astype(np.float32) / 255.0

    lane_frame = detect_lanes(frame)
    lane_frame = cv2.resize(lane_frame, (960, 540))

    # Resize for model input
    img_model = cv2.resize(frame, (input_shape[1], input_shape[0]))
    img_np = np.transpose(img_model, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(img_np, axis=0)
    inputs[0].host = image

    result = common.do_inference(
        context, engine, bindings, inputs, outputs, stream
    )[0]
    result = np.array(result).reshape((5, 5, 10))

    confi_threshold = 0.4
    voting_iou_threshold = 0.5
    bboxs, result_prob = label_to_box_xyxy(result, confi_threshold)

    detected = False
    if len(bboxs) > 0:
        vote_rank = voting_suppression(bboxs, voting_iou_threshold)
        if len(vote_rank) > 0 and result_prob[vote_rank[0]] > confi_threshold:
            bbox = bboxs[vote_rank[0]]
            [c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
            # Scale to 960x540 from model resolution
            scale_x = 960 / input_shape[1]
            scale_y = 540 / input_shape[0]
            c_x *= scale_x
            c_y *= scale_y
            w *= scale_x
            h *= scale_y
            bboxs_scaled = np.array([[c_x, c_y, w, h]])
            lane_frame = draw_bbox(lane_frame.astype(np.float32), bboxs_scaled)

            # Bottom center point
            x_pixel = c_x
            y_pixel = c_y + h / 2
            x_car, y_car = calculate_distance(x_pixel, y_pixel, intrinsic_matrix, mounting_height)
            detected = True
            print(f"Distance to object: x = {x_car:.2f} m, y = {y_car:.2f} m")

    if not detected:
        print("No object detected")

    cv2.imshow("Detection & Lane", lane_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
common.free_buffers(inputs, outputs, stream)
cam.release()
cv2.destroyAllWindows()
