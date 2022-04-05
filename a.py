import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

class Detectron2Parser:
    def __init__(self, segmentation_weight, num_classes, threshold, device):
        self.segmentation_cfg = get_cfg()
        # print(self.segmentation_cfg)
        self.segmentation_cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.segmentation_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.segmentation_cfg.MODEL.WEIGHTS = segmentation_weight
        self.segmentation_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set a custom testing threshold
        if device != 'cpu':
            device = f'cuda:{device}'
        self.segmentation_cfg.MODEL.DEVICE = device
        self.segmentation_predictor = DefaultPredictor(self.segmentation_cfg)

    @staticmethod
    def check_face_location__rotate_segment(image, segment_box, face_box):
        id_x1, id_y1, id_x2, id_y2 = segment_box
        mid_y = (id_y1 + id_y2) / 2
        mid_x = (id_x1 + id_x2) / 2
        f_x1, f_y1, f_x2, f_y2 = face_box
        # print(id_y1, id_y2, id_x1, id_x2)
        roi_image = image[int(id_y1):int(id_y2), int(id_x1):int(id_x2)]
        if (id_x2 - id_x1) < (id_y2 - id_y1):
            if f_y1 < mid_y:
                roi_image = cv2.rotate(roi_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif f_y1 > mid_y:
                roi_image = cv2.rotate(roi_image, cv2.ROTATE_90_CLOCKWISE)
        else:
            if f_x1 > mid_x:
                roi_image = cv2.rotate(roi_image, cv2.ROTATE_180)

        return roi_image

    def segment_single_image(self, image, name):
        outputs = self.segmentation_predictor(image)
        # outputs = predictor(image)
        # print(outputs)
        classes = outputs["instances"].pred_classes.to('cpu').numpy()
        # print(classes)

        if len(classes) >= 2:
            if classes[0] == 0:
                id_card = outputs['instances'].pred_boxes[0].tensor.cpu().numpy()[0]
                face = outputs['instances'].pred_boxes[1].tensor.cpu().numpy()[0]
            elif classes[0] == 1:
                id_card = outputs['instances'].pred_boxes[1].tensor.cpu().numpy()[0]
                face = outputs['instances'].pred_boxes[0].tensor.cpu().numpy()[0]
            # if name:
            #     f_x1, f_y1, f_x2, f_y2 = face
            #     # print(id_y1, id_y2, id_x1, id_x2)
            #     face_image = image[int(f_y1 - (f_y2 - f_y1)/3):int(f_y2 + (f_y2 - f_y1)/2),
            #                  int(f_x1 - (f_x2 - f_x1)/2):int(f_x2 + (f_x2 - f_x1)/2)]
            #     cv2.imwrite(f'data/faces/face_{name}', face_image)
            # print(id_card)
            image_segmentation = self.check_face_location__rotate_segment(image, id_card, face)
            return image_segmentation
        else:
            # TODO : process if len class != 2
            print('len class: ', len(classes))
            return None
        # print(output_predictor)
        # if output_predictor['instances'].pred_masks.shape[0] > 1:
        #     # mask_check = output_predictor['instances'].pred_masks.cpu().numpy()
        #     masks = output_predictor['instances'].pred_masks.cpu().numpy()
        #     # print(mask)
        #     mask_binary = masks[np.argmax(np.sum(masks, axis=(1, 2))), :, :]
        #
        # else:
        #     mask_binary = np.squeeze(output_predictor['instances'].pred_masks.permute(1, 2, 0).cpu().numpy())
        #
        # try:
        #     tol = 0
        #     if mask_binary is None:
        #         mask_binary = image > tol
        #     return image[np.ix_(mask_binary.any(1), mask_binary.any(0))]
        #
        # except ValueError:
        #     print("error")
        #     return None