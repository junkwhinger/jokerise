from PIL import Image

from copy import deepcopy
from inference_JokerCycleGAN import modify_box, jokerise


def inference_frame(frame, face_detector, joker_translater):

    # frame: np array
    target_frame = deepcopy(frame)
    frame_pil = Image.fromarray(frame)
    # 1. extract face
    boxes, probs = face_detector.detect(frame_pil)

    # 2. for each box, translate to joker
    if boxes is None:
        return frame

    for box in boxes:
        sx, sy, ex, ey = box
        # sx = int(sx)
        # sy = int(sy)
        # ex = int(ex)
        # ey = int(ey)
        sx, sy, ex, ey = modify_box(box, 1.0, frame.shape)
        face_patch = frame[sy:ey, sx:ex, :]
        jokerised_face = jokerise(face_patch, joker_translater)

        target_frame[sy:ey, sx:ex, :] = jokerised_face

    return target_frame
