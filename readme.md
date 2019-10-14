# Jokeriser with CycleGAN

![](translated_samples/joaquin.jpg)

Not sure if Joker face would look good on you? Try jokeriser!

Jokeriser finds your face with <a href="https://github.com/timesler/facenet-pytorch">facenet_pytorch</a>  and translate your face to a Joker's using a generator trained with <a href="https://arxiv.org/pdf/1703.10593.pdf">CycleGAN</a>. 

<br>

## Getting Started

```bash
$ git clone https://github.com/junkwhinger/jokerise.git
$ cd jokeriser
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### How to jokerise

```bash
# image
python demo.py --input samples/joaquin.jpg

# video
python demo.py --input samples/joaquin.mp4

# webcam
python demo.py --webcam

# wanna see original & tranlsated version side by side?
python demo.py --webcam --show-original
```

<br>

### Note

- I tested my code in OSX environment. Not sure if it works well on other platforms.

<br>

## CycleGAN Training Details

- Dataset
  - Joaquin Phoenix's and Heath Ledger's joker faces from Google (300 images)
  - Randomly selected faces from CelebA dataset (300 images)
- Preprocessing
  - cropped faces with <a href="https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/">OpenCV</a>
- Training details
  - image size: 3x128x128
  - number of residual blocks: 6
  - epochs: 200

<br>

## Have Fun!

![](translated_samples/joaquin.jpg)

![](translated_samples/joaquin2.jpg)

![](translated_samples/lady.jpg)

![](translated_samples/kim.jpg)

![](translated_samples/joaquin.gif)



## Reference

- Detectron2 https://github.com/facebookresearch/detectron2/tree/master/detectron2
- CycleGAN https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/models
- facenet-pytorch https://github.com/timesler/facenet-pytorch
- Face detection with OpenCV and deep learning https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/