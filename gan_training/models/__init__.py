from gan_training.models import (
    resnet, resnet2, resnet2_longsequence, resnet2_interpolation, resnet2_small, BigGAN
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'resnet2_longsequence': resnet2_longsequence.Generator,
    'resnet2_interpolation': resnet2_interpolation.Generator,
    'resnet2_small': resnet2_small.Generator,
    'biggan': BigGAN.BigGANGenerator
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.ImageDiscriminator,
    'resnet2_longsequence': resnet2_longsequence.ImageDiscriminator,
    'resnet2_interpolation': resnet2_interpolation.ImageDiscriminator,
    'resnet2_small': resnet2_small.ImageDiscriminator
}

video_discriminator_dict = {
    'resnet2': resnet2.VideoDiscriminator,
    'resnet2_longsequence': resnet2_longsequence.VideoDiscriminator,
    'resnet2_interpolation': resnet2_interpolation.VideoDiscriminator,
    'resnet2_small': resnet2_small.VideoDiscriminator
}

text_encoder_dict = {
    'resnet2': resnet2.TextEncoder,
    'resnet2_longsequence': resnet2_longsequence.TextEncoder,
    'resnet2_interpolation': resnet2_interpolation.TextEncoder,
    'resnet2_small': resnet2_small.TextEncoder
}