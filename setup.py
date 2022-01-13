from setuptools import setup

setup(name='jokerise',
      version='0.1.0',
      description='Jokerise',
      author='Junsik Hwang',
      author_email='junkwhinger@me.com',
      url='https://github.com/junkwhinger/jokerise',
      packages=['jokerise'],
      package_data={'jokerise': ['model_weights/e200_net_G_A.pth']},
      python_requires='>=3.6',
      install_requires=[
          'opencv_python_headless==4.1.1.26',
          'torch==1.3.0',
          'facenet_pytorch==0.3.1',
          'numpy==1.17.2',
          'torchvision==0.4.1',
          'Pillow==9.0.0',
          'celluloid==0.2.0',
          'matplotlib==3.1.1',
          'tqdm==4.36.1'
      ])
