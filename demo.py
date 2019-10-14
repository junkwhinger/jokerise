# reference: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import os
import argparse
import numpy as np
import cv2
import tqdm

from predictor import VisualisationDemo
from utils import save_video

def get_parser():
	"""prepare argument parser"""
	parser = argparse.ArgumentParser(description="Jokeriser Demo")
	parser.add_argument("--input", type=str, nargs="*")
	parser.add_argument("--webcam", action="store_true", help="input type: webcam, image, video")

	
	parser.add_argument("--box-multiply-factor", type=float, default=1.1, help="factor to enlarge face bounding box")
	parser.add_argument("--translate", default="jokerise", help="translation type: jokersize / dejokerise")
	parser.add_argument("--in_ch", type=int, default=3, help="number of input channels for CycleGAN generator")
	parser.add_argument("--out_ch", type=int, default=3, help="number of output channels for CycleGAN generator")
	parser.add_argument("--ngf", type=int, default=64, help="number of first conv channels for CycleGAN generator")
	parser.add_argument("--n_blocks", type=int, default=6, help="number of residual blocks for CycleGAN generator")
	parser.add_argument("--img_size", type=int, default=128, help="number of residual blocks for CycleGAN generator")
	parser.add_argument("--generator-weight-path", type=str, default="model_weights/e200_net_G_A.pth", help="model weight file path for CycleGAN generator")

	parser.add_argument("--show-original", action="store_true", help="show original image/video on the left")

	return parser


if __name__ == "__main__":
	args = get_parser().parse_args()
	
	demo = VisualisationDemo(args)

	if args.webcam:
		assert args.input is None
		# webcam
		cam = cv2.VideoCapture(0)
		for orignal_image, translated_image in tqdm.tqdm(demo.run_on_video(cam)):
			if args.show_original:
				concat = np.concatenate([orignal_image, translated_image], axis=1)
			else:
				concat = translated_image
			height, width = concat.shape[:2]
			resized = cv2.resize(concat, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
			cv2.imshow("webcam", resized)
			if cv2.waitKey(1) == 27:
				break #esc to quit

		cam.release()
		cv2.destroyAllWindows()
	
	else:
		# image or videos
		for input_item in args.input:
			fname, ext = os.path.splitext(input_item)

			if ext in [".jpg", ".jpeg", ".png"]:
				original_image = cv2.imread(input_item)
				original_image, translated_image = demo.run_on_image(original_image)

				if args.show_original:
					concat = np.concatenate([original_image, translated_image], axis=1)
				else:
					concat = translated_image

				cv2.imwrite("translated_samples/" + os.path.basename(input_item), concat)

			elif ext in [".mp4", ".avi"]:
				cam = cv2.VideoCapture(input_item)
				fps = cam.get(cv2.CAP_PROP_FPS)

				frames = []
				for orignal_image, translated_image in tqdm.tqdm(demo.run_on_video(cam)):
					if args.show_original:
						concat = np.concatenate([orignal_image, translated_image], axis=1)
					else:
						concat = translated_image
					height, width = concat.shape[:2]
					resized = cv2.resize(concat, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
					frames.append(resized)

				cam.release()

				save_video(frames, "translated_samples/" + os.path.basename(input_item), fps)
				