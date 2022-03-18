import torch.utils.data as data

from PIL import Image
import numpy as np
import os
import os.path
import torch
from datetime import datetime


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def form_vocab(root):
    """
    Function to form text vocabulary
    """
    vocab = {}
    vocab['<eos>'] = 1
    line_list = []
    for fname in os.listdir(root):
        if has_file_allowed_extension(fname, '.txt'):
            file_path = os.path.join(root, fname)
            f = open(file_path, 'r')
            lines = f.readlines()
            for l in lines:
                if l not in line_list:
                    line_list.append(l)

                words = l.split(' ')
                for w in words:
                    w = w.lower()
                    if w not in vocab:
                        vocab[w] = len(vocab)+1

    return vocab, line_list

def form_vocab_list(vocab):
    inv_vocab_dict = {}
    for k in vocab:
        inv_vocab_dict[vocab[k]] = k

    vocab_list = []
    for i in range(len(inv_vocab_dict)):
        vocab_list.append(inv_vocab_dict[i+1])

    return vocab_list

def make_dataset(root, extensions):
    video_root = os.path.join(root, 'videos')
    label_root = os.path.join(root, 'labels')
    text_root = os.path.join(root, 'text_annotations')

    images = []
    # dir = os.path.expanduser(dir)
        
    for fname in sorted(os.listdir(video_root)):
        if has_file_allowed_extension(fname, extensions):
            file_name = fname[:-4]
            lname = file_name + '.txt'
            img_path = os.path.join(video_root, fname)
            text_path = os.path.join(text_root, lname)
            label_path = os.path.join(label_root, lname)
            item = (img_path, text_path, label_path)
            images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, img_loader, text_loader, label_loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        
        samples = make_dataset(root, extensions)
        self.vocab, self.all_lines = form_vocab(os.path.join(root, 'text_annotations'))
        self.vocab_list = form_vocab_list(self.vocab)
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.img_loader = img_loader
        self.text_loader = text_loader
        self.label_loader = label_loader
        self.extensions = extensions
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        (img_path, text_path, label_path) = self.samples[index]
        sample = self.img_loader(img_path)
        text, text_neg = self.text_loader(text_path, self.vocab, self.all_lines)
        label = self.label_loader(label_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, text, text_neg, label

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def sample_neg_line(line, line_list):
    np.random.seed(datetime.now().microsecond)
    index = np.random.randint(len(line_list))
    while line == line_list[index]:
        index = np.random.randint(len(line_list))
    return line_list[index]        


def default_text_loader(path, vocab, line_list):
    f = open(path, 'r')
    lines = f.readlines()

    encoding = []
    encoding_neg = []

    for l in lines:
        neg_line = sample_neg_line(l, line_list) 
        words = l.split(' ')
        for w in words:
            w = w.lower()
            encoding.append(vocab[w])

        words_neg = neg_line.split(' ')
        for w in words_neg:
            w = w.lower()
            encoding_neg.append(vocab[w])

    encoding = np.array(encoding)
    encoding_vec = np.zeros((16))
    encoding_vec[0:len(encoding)] = encoding

    encoding_neg = np.array(encoding_neg)
    encoding_neg_vec = np.zeros((16))
    encoding_neg_vec[0:len(encoding_neg)] = encoding_neg

    return encoding_vec, encoding_neg_vec

def default_label_loader(path):
    f = open(path, 'r')
    line = f.readlines()
    line = line[0]
    tokens = line.split(',')
    label_list = [int(t) for t in tokens]
    label_arr = np.array(label_list)
    return label_arr

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 img_loader=default_loader, text_loader=default_text_loader, label_loader=default_label_loader):
        super(ImageFolder, self).__init__(root, img_loader, text_loader, label_loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
