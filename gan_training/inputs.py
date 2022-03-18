import torch
import gan_training.transforms.transforms as transforms
import gan_training.datasets as datasets
import numpy as np


def get_dataset(name, data_dir, size=64, video_len=16):
    nframes = video_len
    transform = transforms.Compose([
	    transforms.Reshape_vid(nframes),
	    transforms.Resize_vid(size),
	    transforms.ToTensor_vid(nframes),
        transforms.Normalize_vid((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])

    dataset = datasets.ImageFolder(data_dir, transform)
    return dataset


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img/127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img
