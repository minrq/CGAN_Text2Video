import argparse
import os
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.inputs import get_dataset
from gan_training.config import (
    load_config, build_models
)
import subprocess as sp
import numpy as np
import cv2
import torchvision.utils as vutils

# Arguments
parser = argparse.ArgumentParser(
    description='Test a trained GAN and create visualizations.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
checkpoint_dir = path.join(out_dir, 'chkpts')
img_dir = path.join(out_dir, 'paperfig_texture', 'img')
img_gif_dir = path.join(out_dir, 'paperfig_texture', 'img_gif')
label_dir = path.join(out_dir, 'paperfig_texture', 'label')
frame_dir = path.join(out_dir, 'paperfig_texture', 'frames')

def mkdirp(pth):
    if not path.exists(pth):
        os.makedirs(pth)  

# Creat missing directories
if not path.exists(img_dir):
    os.makedirs(img_dir)
if not path.exists(img_gif_dir):
    os.makedirs(img_gif_dir)
if not path.exists(label_dir):
    os.makedirs(label_dir)
if not path.exists(frame_dir):
    os.makedirs(frame_dir)


# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")
video_len = 16
video_len_train = 16

train_dataset = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    video_len=video_len_train
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
)

glove_path = '/net/mlfs01/export/users/ybalaji/Projects/Videogen/code_shapes/deps/GloVe'
glove_dict = utils.get_glove_dict(train_dataset.vocab, glove_path)


generator, image_discriminator, video_discriminator, text_encoder = build_models(config)
print(generator)
print(text_encoder)

# Put models on gpu if needed
generator = generator.to(device)
image_discriminator = image_discriminator.to(device)
video_discriminator = video_discriminator.to(device)
text_encoder = text_encoder.to(device)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    image_discriminator=image_discriminator,
    video_discriminator=video_discriminator,
    text_encoder=text_encoder
)

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

text_encoder = text_encoder.eval()

# Evaluator
evaluator = Evaluator(generator_test, train_loader, glove_dict, batch_size=batch_size, device=device, video_len=video_len)

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')

def save_video_stack(video_tensor, path):

    vid_list = []
    for i in range(video_tensor.size(0)):
        frame = video_tensor[i].permute(1, 2, 0)
        frame = frame.cpu().numpy()
        frame = frame*255
        frame = frame.astype(np.uint8)
        frame = frame[:, :, ::-1]
        vid_list.append(frame)

    vid_stack = np.vstack(vid_list)
    cv2.imwrite(path, vid_stack)

def save_video_gif(ffmpeg, video, filename):
    command = [ffmpeg,
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', '64x64',
               '-pix_fmt', 'rgb24',
               '-r', '8',
               '-i', '-',
               '-c:v', 'gif',
               '-q:v', '3',
               '-an',
               filename]

    try_flag = True
    while try_flag == True:
        try:
            pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
            pipe.stdin.write(video.tostring())
            try_flag = False
        except IOError:
            try_flag = True
            del pipe

def save_frames(video, count):
    # Function to save the individual frames of a video tensor

    nframes = video.size(0)
    folder_path = os.path.join(frame_dir, str(count))
    mkdirp(folder_path)

    for i in range(nframes):
        frame = video[i].unsqueeze(0)
        vutils.save_image(frame, os.path.join(folder_path, '%d.png'%i), padding=0)

# Samples
print('Creating samples...')
num_samples_to_gen = 50
count = 1

inv_vocab_dict = {}
for k in train_dataset.vocab:
    inv_vocab_dict[train_dataset.vocab[k]] = k

for i in range(num_samples_to_gen):
    x, y, y_labels = evaluator.create_samples(text_encoder, batch_size=2, video_len=video_len)
    
    if count%50 == 1:
        print('%d samples created' %count)

    for j in range(x.size(0)):
        vid = x[j].permute(1, 0, 2, 3)
        vid = (vid/2) + 0.5
        
        save_frames(vid, count)
        save_video_stack(vid, os.path.join(img_dir, '%.5d.png'%count))

        vid_gif = vid.permute(0, 2, 3, 1)
        vid_gif = vid_gif.cpu().numpy()
        vid_gif = vid_gif*255
        vid_gif = vid_gif.astype(np.uint8)
        
        save_video_gif('ffmpeg', vid_gif, os.path.join(img_gif_dir, '%.5d.gif' % count))

        label_str = '%d,%d,%d,%d,%d'%(y_labels[j][0], y_labels[j][1], y_labels[j][2], y_labels[j][3], y_labels[j][4])
        f = open(os.path.join(label_dir, '%.5d.txt'%count), 'w')
        f.write(label_str)
        f.close()

        count += 1


