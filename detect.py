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

    overlaps = commonFunctions.compute_polygon_overlaps(ground_truth, out_boxes)
    for parking_area, overlap_areas in zip(ground_truth, overlaps):
        max_IoU_overlap = np.max(overlap_areas)
        pts = np.array(parking_area, np.int32)
        pts = pts.reshape((-1, 1, 2))

        if max_IoU_overlap < 0.25:
            bbox_color = [255, 255, 255]
            bbox_thick = int(0.6 * (original_image.shape[0] + original_image.shape[1]) / 600)
            cv2.polylines(original_image, [pts], True, bbox_color, bbox_thick)
            free_space = True
            count += 1
        else:
            bbox_color = [255, 0, 0]
            bbox_thick = int(0.6 * (original_image.shape[0] + original_image.shape[1]) / 600)
            cv2.polylines(original_image, [pts], True, bbox_color, bbox_thick)

    if free_space:
        free_space_frames += 1
    else:
        free_space_frames = 0

    if free_space_frames > 25 * 5:
        free_space_frames = 0

    image = Image.fromarray(original_image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image, count


if __name__ == '__main__':
    try:
        camera = cv2.VideoCapture('./images/company_1.mp4')

        # frame = cv2.imread('./images/vid1.jpg', 1)
        ground_truth = commonFunctions.get_ground_truth('./data/ground_truth/video_1.p')
        saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                image, count = detect(saved_model_loaded, frame, ground_truth)
                print(count)
                # cv2.imshow("", image)
                cv2.imwrite("test.jpg", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except SystemExit:
        pass
