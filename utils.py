import tensorflow as tf
import numpy as np
import cv2
import math
def non_max_suppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold,
                        confidence_threshold):
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]
    scores = confs * class_probs
    boxes, scores, classes, valid_detections = \
        tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                   tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections
def load_class_names(file_name):
    with open(file_name, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names
def output_boxes(inputs,model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0
    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)
    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size,
                                      max_output_size_per_class, iou_threshold, confidence_threshold)

    return boxes_dicts
def draw_outputs(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)
    for i in range(nums):
        x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)
        img = cv2.putText(img, '{} {:.2f}'.format(
            class_names[int(classes[i])], objectness[i]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    return img

def get_box(boxes, nums, img):
    boxes, nums = boxes[0], nums[0]
    boxes = np.array(boxes)
    for i in range(nums):
        x1y1 = (boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32)
        x2y2 = (boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32)
        
    x1 = x1y1[0]
    y1 = x1y1[1]
    x2 = 0.5*(x2y2[0] + x1)
    y2 = x2y2[1]
    x3 = x2
    y3 = y1
    
    angle_x1_x2 = math.degrees(math.atan2(x2 - x3, y2 - y3))
    angle_x1_x3 = math.degrees(math.atan2(x2 - x1, y2 - y1))
    
    angle_right = 90 + angle_x1_x2
    angle_left = 90 - angle_x1_x3
    
    total_angle = angle_right + angle_left
    
    bench_length = 75

    distance = (bench_length * (1 / total_angle) * 57)
    
    cv2.putText(img, "Distance: {} cm".format(round(distance, 2)), (x1, y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
               (0, 0, 255), 2)
    
    return distance

def get_box2(boxes, nums, img):
    boxes, nums = boxes[0], nums[0]
    boxes = np.array(boxes)
    for i in range(nums):
        x1y1 = (boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32)
        x2y2 = (boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32)
        
    x1 = x1y1[0]
    y1 = x1y1[1]
    x2 = x2y2[0]
    y2 = x2y2[1]
    
    objlength = 136
    focalLength = 26
    sensorHeight = 27.5

    distance = (focalLength*objlength*416)/((y2-y1)*sensorHeight*10)
    
    cv2.putText(img, "Distance: {} cm".format(round(distance, 2)), (x1, y2 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
               (0, 0, 255), 2)
    
    return distance