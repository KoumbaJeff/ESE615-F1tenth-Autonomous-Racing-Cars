# https://github.com/NVIDIA/TensorRT/tree/release/10.3/samples/python/yolov3_onnx

#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import os
import sys

import numpy as np
import tensorrt as trt
from PIL import ImageDraw

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import common

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import time

TRT_LOGGER = trt.Logger()

def draw_bboxes(
    image_raw, bboxes, confidences, categories, all_categories, bbox_color="blue"
):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text(
            (left, top - 12),
            "{0} {1:.2f}".format(all_categories[category], score),
            fill=bbox_color,
        )

    return image_raw


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            0
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 1 << 28
            )  # 256MiB
            # set FP16 or FP32
            config.set_flag(trt.BuilderFlag.FP16)
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                        onnx_file_path
                    )
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 180, 320]
            print("Completed parsing of ONNX file")
            print(
                "Building an engine from file {}; this may take a while...".format(
                    onnx_file_path
                )
            )
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
    
def grid_cell(cell_indx, cell_indy):
    final_dim = [5, 10]
    input_dim = [180, 320]
    anchor_size = [(input_dim[0] / final_dim[0]), (input_dim[1] / final_dim[1])]

    stride_0 = anchor_size[1]
    stride_1 = anchor_size[0]
    return np.array([cell_indx * stride_0, cell_indy * stride_1, cell_indx * stride_0 + stride_0, cell_indy * stride_1 + stride_1])
    
# convert from [c_x, c_y, w, h] to [x_l, y_l, x_r, y_r]
def bbox_convert(c_x, c_y, w, h):
    return [c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2]

# convert from [x_l, y_l, x_r, x_r] to [c_x, c_y, w, h]
def bbox_convert_r(x_l, y_l, x_r, y_r):
    return [x_l/2 + x_r/2, y_l/2 + y_r/2, x_r - x_l, y_r - y_l]
    
def postprocess(result, threshold=0.9):
    validation_result = []
    result_prob = []
    final_dim = [5, 10]
    input_dim = [180, 320]
    anchor_size = [(input_dim[0] / final_dim[0]), (input_dim[1] / final_dim[1])]

    for ind_row in range(final_dim[0]):
        for ind_col in range(final_dim[1]):
            grid_info = grid_cell(ind_col, ind_row)
            validation_result_cell = []
            if result[0, ind_row, ind_col] >= threshold:
                c_x = grid_info[0] + anchor_size[1]/2 + result[1, ind_row, ind_col]
                c_y = grid_info[1] + anchor_size[0]/2 + result[2, ind_row, ind_col]
                w = result[3, ind_row, ind_col] * input_dim[1]
                h = result[4, ind_row, ind_col] * input_dim[0]
                x1, y1, x2, y2 = bbox_convert(c_x, c_y, w, h)
                x1 = np.clip(x1, 0, input_dim[1])
                x2 = np.clip(x2, 0, input_dim[1])
                y1 = np.clip(y1, 0, input_dim[0])
                y2 = np.clip(y2, 0, input_dim[0])
                validation_result_cell.append(x1)
                validation_result_cell.append(y1)
                validation_result_cell.append(x2)
                validation_result_cell.append(y2)
                result_prob.append(result[0, ind_row, ind_col])
                validation_result.append(validation_result_cell)
    validation_result = np.array(validation_result)
    result_prob = np.array(result_prob)
    return validation_result, result_prob

def label_to_box_xyxy(result, threshold = 0.9):
    final_dim = [5, 10]
    input_dim = [180, 320]
    anchor_size = [(input_dim[0] / final_dim[0]), (input_dim[1] / final_dim[1])]
    validation_result = []
    result_prob = []
    for ind_row in range(final_dim[0]):
        for ind_col in range(final_dim[1]):
            grid_info = grid_cell(ind_col, ind_row)
            validation_result_cell = []
            if result[0, ind_row, ind_col] >= threshold:
                c_x = grid_info[0] + anchor_size[1]/2 + result[1, ind_row, ind_col]
                c_y = grid_info[1] + anchor_size[0]/2 + result[2, ind_row, ind_col]
                w = result[3, ind_row, ind_col] * input_dim[1]
                h = result[4, ind_row, ind_col] * input_dim[0]
                x1, y1, x2, y2 = bbox_convert(c_x, c_y, w, h)
                x1 = np.clip(x1, 0, input_dim[1])
                x2 = np.clip(x2, 0, input_dim[1])
                y1 = np.clip(y1, 0, input_dim[0])
                y2 = np.clip(y2, 0, input_dim[0])
                validation_result_cell.append(x1)
                validation_result_cell.append(y1)
                validation_result_cell.append(x2)
                validation_result_cell.append(y2)
                result_prob.append(result[0, ind_row, ind_col])
                validation_result.append(validation_result_cell)
    validation_result = np.array(validation_result)
    result_prob = np.array(result_prob)
    return validation_result, result_prob


def voting_suppression(result_box, iou_threshold = 0.5):
    votes = np.zeros(result_box.shape[0])
    for ind, box in enumerate(result_box):
        for box_validation in result_box:
            if IoU(box_validation, box) > iou_threshold:
                votes[ind] += 1
    return (-votes).argsort()

def IoU(a, b):
    # referring to IoU algorithm in slides
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter_ab = inter_w * inter_h
    area_a = (a[3] - a[1]) * (a[2] - a[0])
    area_b = (b[3] - b[1]) * (b[2] - b[0])
    union_ab = area_a + area_b - inter_ab
    return inter_ab / union_ab

# def DisplayLabel(img, bboxs):
#     # image = np.transpose(image.copy(), (1, 2, 0))
#     # fig, ax = plt.subplots(1, figsize=(6, 8))
#     image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
#     fig, ax = plt.subplots(1)
#     edgecolor = [1, 0, 0]
#     if len(bboxs) == 1:
#         bbox = bboxs[0]
#         ax.add_patch(patches.Rectangle((bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2), bbox[2], bbox[3], linewidth=1, edgecolor=edgecolor, facecolor='none'))
#     elif len(bboxs) > 1:
#         for bbox in bboxs:
#             ax.add_patch(patches.Rectangle((bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2), bbox[2], bbox[3], linewidth=1, edgecolor=edgecolor, facecolor='none'))
#     ax.imshow(image)
#     # plt.show()
#     return image

def DisplayLabel(img, bboxs, window_name="YOLO Output"):
    image = img.copy()
    for bbox in bboxs:
        x, y, w, h = bbox
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)  # Red box

    while True:
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    # Paths
    onnx_file_path = "f1tenth_model.onnx"
    engine_file_path = "f1tenth_model.trt"
    # img_path = "resource/test_car_x60cm.png"
    # img_path = "1000.jpg"
    img_path = "1781.jpg"
    output_image_path = "output_with_boxes.png"

    input_shape = (180, 320)  # (H, W)

    img = cv2.imread(img_path).astype(np.float32) / 255.0
    img_orig = cv2.resize(img, (960, 540))
    img = cv2.resize(img, (input_shape[1], input_shape[0]))  # (W, H)
    img_np = np.transpose(img, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(img_np, axis=0)  # Shape: (1, 3, H, W)

    # Build engine and run inference
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = image  # Preprocessed image

        print(f"Running inference on {img_path}")

        # === Inference timing block ===
        NUM_RUNS = 100
        # Warm-up
        for _ in range(10):
            _ = common.do_inference(context, engine, bindings, inputs, outputs, stream)

        # Timing
        times = []
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            result = common.do_inference(context, engine, bindings, inputs, outputs, stream)[0]
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_time = sum(times) / len(times)
        print(f"Average inference time over {NUM_RUNS} runs: {avg_time:.2f} ms")
        # ==============================

    # Reshape output
    print("Raw output shape:", np.array(result).shape)
    result = np.array(result).reshape((5, 5, 10))

    # Postprocessing
    voting_iou_threshold = 0.5
    confi_threshold = 0.4
    bboxs, result_prob = label_to_box_xyxy(result, confi_threshold)
    
    vote_rank = voting_suppression(bboxs, voting_iou_threshold)
    bbox = bboxs[vote_rank[0]]
    [c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
    bboxs_2 = np.array([[c_x, c_y, w, h]])

    # Scale to original resolution
    scale_x = 960 / input_shape[1]
    scale_y = 540 / input_shape[0]
    c_x *= scale_x
    c_y *= scale_y
    w *= scale_x
    h *= scale_y
    bboxs_scaled = np.array([[c_x, c_y, w, h]])

    # Visualize result
    DisplayLabel(img_orig.astype(np.float32), bboxs_scaled)

    common.free_buffers(inputs, outputs, stream)


if __name__ == "__main__":
    main()