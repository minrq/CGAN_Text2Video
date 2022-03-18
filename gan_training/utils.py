
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import numpy as np
import os


def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow, padding=0)

def save_video(video, outfile, nrow=8):
    video = video / 2 + 0.5     # unnormalize
    video = video.permute(0, 2, 1, 3, 4).contiguous()
    video = video.view(video.size(0)*video.size(1), video.size(2), video.size(3), video.size(4))
    torchvision.utils.save_image(video, outfile, nrow=nrow)

def get_nsamples(data_loader, N):
    x = []
    t = []
    l = []
    n = 0
    while n < N:
        x_next, t_next, __, l_next = next(iter(data_loader))
        x.append(x_next)
        t.append(t_next)
        l.append(l_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    t = torch.cat(t, dim=0)[:N]
    l = torch.cat(l, dim=0)[:N]

    return x, t, l


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def get_glove_dict(vocab, glove_path, ndim=300):

    num_vocab = len(vocab)
    embedding_tensor = torch.FloatTensor(num_vocab+1, ndim).zero_()

    with open(os.path.join(glove_path, 'glove.42B.300d.txt'), 'r') as f:
        for line in f:
            line_split = line.split(' ')
            word = line_split[0]
            if word not in vocab:
                continue
            
            embedding_vector = [float(l) for l in line_split[1:]]
            embedding_vector = np.array(embedding_vector)
            embedding_vector = torch.from_numpy(embedding_vector)
            index = vocab[word]
            embedding_tensor[index] = embedding_vector

    return embedding_tensor 


def get_glove_embedding(glove_dict, y_encoding, ndim=300):
    bs = y_encoding.size(0)
    num_words = y_encoding.size(1)
    emb = torch.FloatTensor(bs, num_words, ndim).zero_()

    for i in range(bs):
        emb[i] = glove_dict[y_encoding[i], :]

    emb = emb.permute(0, 2, 1)
    return emb

