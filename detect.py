import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from commons import commonFunctions

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.functions import *
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from imutils.video import WebcamVideoStream


def detect(saved_model_loaded, original_image, ground_truth):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    allowed_classes = ['car', 'motorbike', 'truck']

    out_boxes, out_scores, out_classes, num_boxes = pred_bbox
    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)

    count = 0
    free_space_frames = 0
    free_space = False

    # car_boxes = []
    # for i in range(num_boxes):
    # if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
    # coor = out_boxes[0][i]
    # class_ind = int(out_classes[0][i])
    # class_name = classes[class_ind]
    #
    # if class_name not in allowed_classes:
    #     continue
    # else:
    #     car_boxes.append(coor)
    #
    # print(car_boxes)
    overlaps = commonFunctions.compute_polygon_overlaps(ground_truth, out_boxes)
    vacant = set()
    fontScale = 0.5
    for index, (parking_area, overlap_areas) in enumerate(zip(ground_truth, overlaps)):

        y1, x1, y2, x2 = parking_area
        max_IoU_overlap = np.max(overlap_areas)
        pts = np.array(parking_area, np.int32)
        pts = pts.reshape((-1, 1, 2))

        if max_IoU_overlap < 0.25:
            label = "VT" + str(index + 1)
            bbox_color = [255, 255, 255]
            bbox_thick = int(0.6 * (original_image.shape[0] + original_image.shape[1]) / 600)
            cv2.polylines(original_image, [pts], True, bbox_color, bbox_thick)

            # t_size = cv2.getTextSize(label, 0, fontScale, thickness=bbox_thick // 2)[0]
            # c1 = (x2,y2)
            # c3 = (x2[0] + t_size[0], x2[1] - t_size[1] - 3)
            # cv2.rectangle(original_image, c1, c3, bbox_color, -1)  # filled
            cv2.putText(original_image, label, (x2[0], np.float32(x2[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            free_space = True
            count += 1
            vacant.add(index + 1)
        else:
            label = "VT" + str(index + 1)
            bbox_color = [255, 0, 0]
            bbox_thick = int(0.6 * (original_image.shape[0] + original_image.shape[1]) / 600)
            cv2.polylines(original_image, [pts], True, bbox_color, bbox_thick)
            # t_size = cv2.getTextSize(label, 0, fontScale, thickness=bbox_thick // 2)[0]
            # c1 = (x2, y2)
            # c3 = (x2[0] + t_size[0], x2[1] - t_size[1] - 3)
            # cv2.rectangle(original_image, c1, c3, bbox_color, -1)  # filled
            cv2.putText(original_image, label, (x2[0], np.float32(x2[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    if free_space:
        free_space_frames += 1
    else:
        free_space_frames = 0

    if free_space_frames > 25 * 5:
        free_space_frames = 0

    image = Image.fromarray(original_image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image, count, vacant


def get_vacant(saved_model_loaded, original_image, ground_truth):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]


    out_boxes, out_scores, out_classes, num_boxes = pred_bbox
    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)

    count = 0
    free_space_frames = 0
    free_space = False

    # car_boxes = []
    # for i in range(num_boxes):
    # if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
    # coor = out_boxes[0][i]
    # class_ind = int(out_classes[0][i])
    # class_name = classes[class_ind]
    #
    # if class_name not in allowed_classes:
    #     continue
    # else:
    #     car_boxes.append(coor)
    #
    # print(car_boxes)
    overlaps = commonFunctions.compute_polygon_overlaps(ground_truth, out_boxes)
    vacant = set()
    for index, (parking_area, overlap_areas) in enumerate(zip(ground_truth, overlaps)):

        max_IoU_overlap = np.max(overlap_areas)
        pts = np.array(parking_area, np.int32)

        if max_IoU_overlap < 0.25:
            vacant.add(index + 1)
    return vacant


if __name__ == '__main__':
    try:
        camera = WebcamVideoStream('./images/company_1.mp4')
        ground_truth = commonFunctions.get_ground_truth('./data/ground_truth/video_1.p')
        saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        while True:
            frame = camera.read()

            image, count, index = detect(saved_model_loaded, frame, ground_truth)
            cv2.waitKey(1)
            cv2.imshow("", image)
            # cv2.imwrite("test.jpg", image)

        cv2.destroyAllWindows()

        # frame = cv2.imread('./images/com.png', 1)
        # image, count = detect(saved_model_loaded, frame, ground_truth)
        # print(count)
        # cv2.imshow("", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except SystemExit:
        pass
