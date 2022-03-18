import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable

from gan_training.models import layers
from gan_training.models.sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d
import pdb

def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch


# def biggan_generator(z_dim, size, **kwargs):
#     G = functools.partial(Generator.__init__, dim_z=z_dim, dim_f=2*z_dim, resolution=size, G_ch=32, G_attn='32',
#                           video_len=kwargs['video_len'])
#     return G


class Generator(nn.Module):
  def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000, dim_f=128,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn', video_len=16,
               **kwargs):
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Kernel size?
    self.kernel_size = G_kernel_size
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    self.dim_f = dim_f
    # Use shared embeddings?
    self.G_shared = G_shared
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    # Hierarchical latent space?
    self.hier = hier
    # Cross replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.G_param = G_param
    # Normalization style
    self.norm_style = norm_style
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # fp16?
    self.fp16 = G_fp16
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[resolution]

    # Recurrent noise generation
    self.recurrent = nn.GRUCell(2*self.dim_z, self.dim_z)
    self.video_length = video_len

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    # self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else layers.identity)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.dim_f + self.z_chunk_size if self.G_shared
                                      else self.dim_f),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)

    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    # self.shared = (self.which_embedding(self.dim_f, self.shared_dim) if G_shared 
    #                else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      from gan_training.models import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z_):
    video_len = self.video_length
    y, z = torch.split(z_, [self.dim_f, self.dim_z], dim=1)
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
    
    # Apply batchnorm-relu-conv-tanh at output
    out = torch.tanh(self.output_layer(h))
    out = out.view(int(out.size(0) / video_len), video_len, 3, out.size(2), out.size(3))
    out = out.permute(0, 2, 1, 3, 4)
    return out
  
  def sample_z_m(self, num_samples, y_mot, video_len=None, device=None):
    video_len = video_len if video_len is not None else self.video_length
    h_t = [self.get_gru_initial_state(num_samples, device)]
    for frame_num in range(video_len):
      e_t = torch.cat((self.get_iteration_noise(num_samples, device), y_mot), 1)
      h_t.append(self.recurrent(e_t, h_t[-1]))
    z_m_t = [h_k.view(-1, 1, self.z_dim) for h_k in h_t]
    z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.z_dim)
    return z_m

  def sample_z_content(self, num_samples, y_cont, video_len=None, device=None):
    video_len = video_len if video_len is not None else self.video_length
    content = np.random.normal(0, 1, (num_samples, self.z_dim)).astype(np.float32)
    content = np.repeat(content, video_len, axis=0)
    content = torch.from_numpy(content)
    y_cont_exp = torch.unsqueeze(y_cont, 1).expand(-1, video_len, -1).contiguous()
    y_cont_exp = y_cont_exp.view(y_cont_exp.size(0)*y_cont_exp.size(1), y_cont_exp.size(2))
    content = content.to(device)
    content = torch.cat((content, y_cont_exp), 1)
    return content

  def sample_z_video(self, num_samples, y_cont, y_mot, video_len=None, device=None):
    video_len = video_len if video_len is not None else self.video_length
    z_content = self.sample_z_content(num_samples, y_cont, video_len, device=device)
    z_motion = self.sample_z_m(num_samples, y_mot, video_len, device=device)
    z = torch.cat([z_content, z_motion], dim=1)
    return z

  def get_gru_initial_state(self, num_samples, device):
    return Variable(torch.FloatTensor(num_samples, self.z_dim).normal_().to(device))

  def get_iteration_noise(self, num_samples, device):
    return Variable(torch.FloatTensor(num_samples, self.z_dim).normal_().to(device))


# wrapper class
class BigGANGenerator(Generator):
  def __init__(self, z_dim=256, size=64, **kwargs):
    super(BigGANGenerator, self).__init__(dim_z=z_dim, dim_f=2*z_dim, resolution=size, G_ch=32, G_attn='32',
          **kwargs)
    self.z_dim = z_dim