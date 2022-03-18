import argparse
import os
from os import path
import time
import copy
import torch
from torch import nn
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)
import pdb
# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Short hands
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']
sample_nlabels = config['training']['sample_nlabels']
video_len = config['training']['video_len']
dir_skipthoughts = '../data/deps/skip-thoughts'

out_dir = config['training']['out_dir']
checkpoint_dir = path.join(out_dir, 'chkpts')

# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")


# Dataset
train_dataset = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    video_len=video_len
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
)

# Number of labels
nlabels = 1
sample_nlabels = min(nlabels, sample_nlabels)

# Create models
generator, image_discriminator, video_discriminator, text_encoder = build_models(config)
print(generator)
print(image_discriminator)
print(video_discriminator)
print(text_encoder)

# Put models on gpu if needed
generator = generator.to(device)
image_discriminator = image_discriminator.to(device)
video_discriminator = video_discriminator.to(device)
text_encoder = text_encoder.to(device)

g_optimizer, d_optimizer, vd_optimizer, enc_optimizer = build_optimizers(
    generator, image_discriminator, video_discriminator, text_encoder, config
)

# Use multiple GPUs if possible
"""
generator = nn.DataParallel(generator)
image_discriminator = nn.DataParallel(image_discriminator)
video_discriminator = nn.DataParallel(video_discriminator)
"""

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    image_discriminator=image_discriminator,
    video_discriminator=video_discriminator,
    text_encoder=text_encoder,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    vd_optimizer=vd_optimizer,
    enc_optimizer=enc_optimizer
)

# Logger
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
)

# Save for tests
ntest = batch_size
x_real, __, __ = utils.get_nsamples(train_loader, ntest)
utils.save_video(x_real, path.join(out_dir, 'real.png'))

# Text modules
glove_path = '../data/deps/GloVe'
glove_dict = utils.get_glove_dict(train_dataset.vocab, glove_path)


# Test generator
if config['training']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Evaluator
evaluator = Evaluator(generator_test, train_loader, glove_dict, batch_size=batch_size, device=device)

# Train
tstart = t0 = time.time()
it = epoch_idx = -1

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')
if it != -1:
    logger.load_stats('stats.p')

# Reinitialize model average if needed
if (config['training']['take_model_average']
        and config['training']['model_average_reinit']):
    update_average(generator_test, generator, 0.)

# Learning rate anneling
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)
vd_scheduler = build_lr_scheduler(vd_optimizer, config, last_epoch=it)
enc_scheduler = build_lr_scheduler(enc_optimizer, config, last_epoch=it)

# Trainer
trainer = Trainer(
    generator, image_discriminator, video_discriminator, text_encoder, g_optimizer, d_optimizer, vd_optimizer, enc_optimizer,
    gan_type=config['training']['gan_type'],
    reg_type=config['training']['reg_type'],
    img_reg_param=config['training']['img_reg_param'],
    vid_reg_param=config['training']['vid_reg_param'],
    batch_size=config['training']['batch_size'],
    video_len=config['training']['video_len'],
    device=device
)

# Training loop
print('Start training...')
while True:
    epoch_idx += 1
    print('Start epoch %d...' % epoch_idx)

    for x_real, y_real, y_neg, __ in train_loader:
        
        it += 1
        g_scheduler.step()
        d_scheduler.step()
        vd_scheduler.step()

        d_lr = d_optimizer.param_groups[0]['lr']
        g_lr = g_optimizer.param_groups[0]['lr']
        vd_lr = vd_optimizer.param_groups[0]['lr']
        logger.add('learning_rates', 'image_discriminator', d_lr, it=it)
        logger.add('learning_rates', 'video_discriminator', vd_lr, it=it)
        logger.add('learning_rates', 'generator', g_lr, it=it)

        y_real = y_real.long()
        y_neg = y_neg.long()

        y_encoding = utils.get_glove_embedding(glove_dict, y_real)
        y_encoding_neg = utils.get_glove_embedding(glove_dict, y_neg)

        x_real = x_real.to(device)
        y_encoding = y_encoding.to(device)
        y_encoding_neg = y_encoding_neg.to(device)

        # Discriminator updates
        dloss, reg = trainer.image_discriminator_trainstep(x_real, y_encoding, y_encoding_neg)
        logger.add('losses', 'image_discriminator', dloss, it=it)
        logger.add('losses', 'image_regularizer', reg, it=it)

        dloss, reg = trainer.video_discriminator_trainstep(x_real, y_encoding, y_encoding_neg)
        logger.add('losses', 'video_discriminator', dloss, it=it)
        logger.add('losses', 'video_regularizer', reg, it=it)

        # Generators updates
        if ((it + 1) % d_steps) == 0:
            gloss = trainer.generator_trainstep(y_encoding)
            logger.add('losses', 'generator', gloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test, generator,
                               beta=config['training']['model_average_beta'])

        # Print stats
        g_loss_last = logger.get_last('losses', 'generator')
        d_loss_last = logger.get_last('losses', 'image_discriminator')
        vd_loss_last = logger.get_last('losses', 'video_discriminator')
        d_reg_last = logger.get_last('losses', 'image_regularizer')
        vd_reg_last = logger.get_last('losses', 'video_regularizer')
        print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, vd_loss = %.4f, img_reg=%.4f, vid_reg=%.4f'
              % (epoch_idx, it, g_loss_last, d_loss_last, vd_loss_last, d_reg_last, vd_reg_last))

        # (i) Sample if necessary
        if (it % config['training']['sample_every']) == 0:
            print('Creating samples...')
            x, y, __ = evaluator.create_samples(text_encoder)
            logger.add_video(x, 'all', it)
            

        # (iii) Backup if necessary
        if ((it + 1) % backup_every) == 0:
            print('Saving backup...')
            checkpoint_io.save(it, 'model_%08d.pt' % it)
            logger.save_stats('stats_%08d.p' % it)

        # (iv) Save checkpoint if necessary
        if time.time() - t0 > save_every:
            print('Saving checkpoint...')
            checkpoint_io.save(it, 'model.pt')
            logger.save_stats('stats.p')
            t0 = time.time()

            if (restart_every > 0 and t0 - tstart > restart_every):
                exit(3)
