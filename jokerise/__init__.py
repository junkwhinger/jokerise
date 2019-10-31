from pkg_resources import resource_filename

from jokerise.predictor import FaceTranslator


def create_jokeriser(in_ch=3,
                     out_ch=3,
                     ngf=64,
                     n_blocks=6,
                     img_size=128,
                     generator_weight_path=resource_filename(
                         __package__, "model_weights/e200_net_G_A.pth"),
                     box_multiply_factor=1.1):

    class JokeriseArgs:

        def __init__(self):
            # Number of input channels for CycleGAN generator
            self.in_ch = in_ch
            # Number of output channels for CycleGAN generator
            self.out_ch = out_ch
            # Number of first conv channels for CycleGAN generator
            self.ngf = ngf
            # Number of residual blocks for CycleGAN generator
            self.n_blocks = n_blocks
            # Number of residual blocks for CycleGAN generator
            self.img_size = img_size
            # Model weight file path for CycleGAN generator
            self.generator_weight_path = generator_weight_path
            # Factor to enlarge face bounding box
            self.box_multiply_factor = box_multiply_factor

    args = JokeriseArgs()
    jokeriser = FaceTranslator(args)
    return jokeriser
