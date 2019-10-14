from copy import deepcopy
from PIL import Image

import cv2
import torch
import torchvision.transforms as tvtransforms
from torchvision.utils import make_grid

from facenet_pytorch import MTCNN
from network import Generator

class FaceDetector:

    def __init__(self, cfg):
        self.face_detector = MTCNN()
        self.box_multiply_factor = cfg.box_multiply_factor

    def __call__(self, original_image):
        """
        Args:
            original_image: numpy array

        """
        frame_info = original_image.shape

        original_image_pil = Image.fromarray(original_image)
        boxes, probs = self.face_detector.detect(original_image_pil)

        if boxes is None:
            return []

        if self.box_multiply_factor != 1.0:
            _tmp = []
            for box in boxes:
                _tmp.append(self.modify_box(box, self.box_multiply_factor, frame_info))
            face_boxes = _tmp
        else:
            face_boxes = boxes
        
        return face_boxes

    def modify_box(self, box_coord, mult_factor, frame_info):
        startX, startY, endX, endY = box_coord
            
        width = endX - startX
        height = endY - startY
        
        max_side = int(max(width, height) * mult_factor)
        half_diameter = int(max_side / 2.0)
        
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        
        startX_new = cX - half_diameter
        startY_new = cY - half_diameter
        endX_new = cX + half_diameter
        endY_new = cY + half_diameter
        
        h, w, c = frame_info
        startX_new = max(0, startX_new)
        startY_new = max(0, startY_new)
        endX_new = min(w, endX_new)
        endY_new = min(h, endY_new)

        return [startX_new, startY_new, endX_new, endY_new]


class Translator:

    def __init__(self, cfg):
        self.translator = Generator(cfg.in_ch, cfg.out_ch, cfg.ngf, cfg.n_blocks)
        self.transform_fn = self.get_transform_fn(cfg.img_size)

        self.translator.load_state_dict(torch.load(cfg.generator_weight_path, map_location="cpu"))

    def __call__(self, image):
        input_pil = Image.fromarray(image[:, :, ::-1])

        input_tensor = self.transform_fn(input_pil).unsqueeze(0)

        with torch.no_grad():
            translated_tensor = self.translator(input_tensor)
            translated_image = make_grid(translated_tensor.cpu(), normalize=True).numpy().transpose(1, 2, 0)[:,:,::-1] * 255.
            
            height, width, _ = image.shape
            resized_image = cv2.resize(translated_image, (height, width))
        
        return resized_image

    def get_transform_fn(self, img_size):
        transform_fn = tvtransforms.Compose([
            tvtransforms.Resize(img_size, Image.BICUBIC),
            tvtransforms.ToTensor(),
            tvtransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transform_fn


class FaceTranslator:

    def __init__(self, cfg):
        self.face_model = FaceDetector(cfg)
        self.translator_model = Translator(cfg)
        self.original_image = None
        self.face_boxes = None

    def __call__(self, original_image):
        self.original_image = original_image
        self.translated_image = deepcopy(original_image)
        self.face_boxes = self.face_model(original_image)

        if len(self.face_boxes) >= 1:
            for box in self.face_boxes:
                sx, sy, ex, ey = box
                face_patch = self.original_image[sy:ey, sx:ex, :]
                translated_patch = self.translator_model(face_patch)

                if translated_patch.shape[:2] != (ey-sy, ex-sx):
                    return self.translated_image
                self.translated_image[sy:ey, sx:ex, :] = translated_patch

        return self.translated_image


class VisualisationDemo(object):
    def __init__(self, cfg):
        self.predictor = FaceTranslator(cfg)
        
    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_image(self, image):
        """
        Args:
            image (cv2.imread)
        """
        return image, self.predictor(image)


    def run_on_video(self, video):
        """
        Args:
            video (cv2.VideoCapture)
        """
        frame_gen = self._frame_from_video(video)

        for frame in frame_gen:
            yield frame, self.predictor(frame)
