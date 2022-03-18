import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim, size, embed_size=64, nfilter=32, **kwargs):
        super(Generator, self).__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.z_dim = z_dim
        self.video_length = 16

        # Recurrent noise generation
        self.recurrent = nn.GRUCell(2*self.z_dim, self.z_dim)
        
        # Submodules
        # self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(3*z_dim, 8*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(8*nf, 8*nf)
        self.resnet_1_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_3_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)


    def forward(self, z):
        
        # z here is the concatenated vector -- the first dimension is batch size x video length

        video_len = self.video_length
        out = self.fc(z)
        batch_size = out.size(0)
        out = out.view(batch_size, 8*self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out)
        
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_1_0(out)
        
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_2_0(out)
        
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_3_0(out)
        
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_4_0(out)
        
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_5_0(out)
        
        out = self.conv_img(actvn(out))
        out = F.tanh(out)

        # The resulting image tensor needs to be reshaped to video tensor
        out = out.view(out.size(0) / video_len, video_len, 3, out.size(2), out.size(3))
        out = out.permute(0, 2, 1, 3, 4)

        return out

    def sample_z_m(self, num_samples, y_mot1, y_mot2, video_len=None, device=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples, device)]

        for frame_num in range(video_len):
            alpha = float(frame_num)/(video_len-1)
            y_mot = (1-alpha)*y_mot1 + alpha*y_mot2
            e_t = torch.cat((self.get_iteration_noise(num_samples, device), y_mot), 1)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.z_dim) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.z_dim)

        return z_m

    
    def sample_z_content(self, num_samples, y_cont1, y_cont2, video_len=None, device=None):
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.z_dim)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)

        alpha_list = [float(i)/(video_len-1) for i in range(video_len)]
        
        # print(y_cont1.size())
        # print(y_cont2.size())
        
        y_cont_exp = [((1-alpha)*y_cont1 + alpha*y_cont2).unsqueeze(1) for alpha in alpha_list]
        y_cont_exp = torch.cat(y_cont_exp, dim=1)
        y_cont_exp = y_cont_exp.view(y_cont_exp.size(0)*y_cont_exp.size(1), y_cont_exp.size(2))

        content = content.to(device)
        content = torch.cat((content, y_cont_exp), 1)
        return content

    def sample_z_video(self, num_samples, y_cont1, y_mot1, y_cont2, y_mot2, video_len=None, device=None):
        
        """
        y_cont1 = y_cont1.unsqueeze(0)
        y_cont2 = y_cont2.unsqueeze(0)
        y_mot1 = y_mot1.unsqueeze(0)
        y_mot2 = y_mot2.unsqueeze(0)
        """

        z_content = self.sample_z_content(num_samples, y_cont1, y_cont2, video_len, device=device)
        z_motion = self.sample_z_m(num_samples, y_mot1, y_mot2, video_len, device=device)
        
        z = torch.cat([z_content, z_motion], dim=1)
        return z

    def get_gru_initial_state(self, num_samples, device):
        return Variable(torch.FloatTensor(num_samples, self.z_dim).normal_().to(device))

    def get_iteration_noise(self, num_samples, device):
        return Variable(torch.FloatTensor(num_samples, self.z_dim).normal_().to(device))



class ImageDiscriminator(nn.Module):
    def __init__(self, z_dim, size, embed_size=64, nfilter=32, **kwargs):
        super(ImageDiscriminator, self).__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.nemb = 128
        
        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1*nf, 2*nf)
        self.resnet_1_0 = ResnetBlock(2*nf, 4*nf)
        self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_3_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_4_0 = ResnetBlock(4*nf, 8*nf)
        self.resnet_5_0 = ResnetBlock(8*nf, 8*nf)
        self.fc = nn.Sequential(
            nn.Linear(3*self.nemb, 1)
        )

        # stage nets
        self.stage1_in_transform = nn.Conv2d(4*nf, nf, 1, 1, 0)
        self.stage2_in_transform = nn.Conv2d(4*nf, nf, 1, 1, 0)

        self.stage1_net = nn.Sequential(
            nn.AvgPool2d(4, 4),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(4, 4),
            nn.Conv2d(nf, self.nemb, 2, 1, 0)
        )

        self.stage2_net = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(nf, self.nemb, 2, 1, 0)
        )

        self.stage3_net = nn.Sequential(
            nn.Linear(8*nf*s0*s0 + z_dim, self.nemb)
        )
        

    def forward(self, x, y_filt, y_encoding):
        
        # Converting videos to images
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        video_len = x.size(1)
        x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        batch_size = x.size(0)

        y_filt_ex = y_filt.unsqueeze(1).expand(-1, video_len, -1, -1, -1, -1).contiguous()
        y_filt_ex = y_filt_ex.view(y_filt_ex.size(0)*y_filt_ex.size(1), y_filt_ex.size(2), y_filt_ex.size(3), y_filt_ex.size(4), y_filt_ex.size(5))
        y_filt_stage1 = y_filt_ex[:, 0:self.nf, :, :, :]
        y_filt_stage2 = y_filt_ex[:, self.nf:, :, :, :]

        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet_0_0(out)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        
        # Stage 1
        out_s1 = self.stage1_in_transform(out)
        out_s1 = [F.conv2d(out_s1[i].unsqueeze(0), y_filt_stage1[i], stride=1, padding=2) for i in range(batch_size)]
        out_s1 = torch.cat(out_s1)
        out_s1 = self.stage1_net(out_s1)
        out_s1 = out_s1.view(batch_size, self.nemb)


        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        
        # Stage 2
        out_s2 = self.stage2_in_transform(out)
        out_s2 = [F.conv2d(out_s2[i].unsqueeze(0), y_filt_stage2[i], stride=1, padding=2) for i in range(batch_size)]
        out_s2 = torch.cat(out_s2)
        out_s2 = self.stage2_net(out_s2)
        out_s2 = out_s2.view(batch_size, self.nemb)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        
        # Stage 3
        out = out.view(batch_size, 8*self.nf*self.s0*self.s0)
        y_encoding_expanded = y_encoding.unsqueeze(1).expand(-1, video_len, -1).contiguous()
        y_encoding_expanded = y_encoding_expanded.view(y_encoding_expanded.size(0)*y_encoding_expanded.size(1), y_encoding_expanded.size(2))
        out_conditioned = torch.cat((out, y_encoding_expanded), 1)
        out_s3 = self.stage3_net(actvn(out_conditioned))

        out = torch.cat((out_s1, out_s2, out_s3), 1)
        out = self.fc(actvn(out))

        return out

class VideoDiscriminator(nn.Module):
    def __init__(self, z_dim, size, embed_size=64, nfilter=32, **kwargs):
        super(VideoDiscriminator, self).__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.nemb = 128
        
        # Submodules
        self.conv_img = nn.Conv3d(3, 1*nf, 3, padding=1)

        self.resnet_0_0 = VideoResnetBlock(1*nf, 2*nf)
        self.resnet_1_0 = VideoResnetBlock(2*nf, 4*nf)
        self.resnet_2_0 = VideoResnetBlock(4*nf, 4*nf)
        self.resnet_3_0 = VideoResnetBlock(4*nf, 4*nf)
        self.resnet_4_0 = VideoResnetBlock(4*nf, 8*nf)
        self.resnet_5_0 = VideoResnetBlock(8*nf, 8*nf)
        
        self.fc = nn.Sequential(
            nn.Linear(3*self.nemb, 1)
        )

        # stage nets
        self.stage1_in_transform = nn.Conv3d(4*nf, nf, 1, 1, 0)
        self.stage2_in_transform = nn.Conv3d(4*nf, nf, 1, 1, 0)

        self.stage1_net = nn.Sequential(
            nn.AvgPool3d((2, 4, 4), (2, 4, 4)),
            nn.Conv3d(nf, nf, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool3d(4, 4),
            nn.Conv3d(nf, self.nemb, (1, 2, 2), 1, 0)
        )

        self.stage2_net = nn.Sequential(
            nn.AvgPool3d(2, 2),
            nn.Conv3d(nf, nf, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool3d(2, 2),
            nn.Conv3d(nf, self.nemb, (1, 2, 2), 1, 0)
        )

        self.stage3_net = nn.Sequential(
            nn.Linear(8*nf*s0*s0 + z_dim, self.nemb)
        )

        self.downscale = nn.AvgPool3d(3, stride=2, padding=1)
        self.downscale_v2 = nn.AvgPool3d((2, 3, 3), stride=2, padding=(0, 1, 1))
        self.downscale_spatial = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))


    def forward(self, x, y_filt, y_encoding):
        batch_size = x.size(0)

        y_filt_stage1 = y_filt[:, 0:self.nf, :, :, :, :]
        y_filt_stage2 = y_filt[:, self.nf:, :, :, :, :]

        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        
        out = self.downscale(out)
        out = self.resnet_1_0(out)

        # Stage 1
        out_s1 = self.stage1_in_transform(out)
        out_s1 = [F.conv3d(out_s1[i].unsqueeze(0), y_filt_stage1[i], stride=1, padding=(1, 2, 2)) for i in range(batch_size)]
        out_s1 = torch.cat(out_s1)
        out_s1 = self.stage1_net(out_s1)
        out_s1 = out_s1.view(batch_size, self.nemb)
        
        out = self.downscale_spatial(out)
        out = self.resnet_2_0(out)
        
        out = self.downscale(out)
        out = self.resnet_3_0(out)

        # Stage 2
        out_s2 = self.stage2_in_transform(out)
        out_s2 = [F.conv3d(out_s2[i].unsqueeze(0), y_filt_stage2[i], stride=1, padding=2) for i in range(batch_size)]
        out_s2 = torch.cat(out_s2)
        out_s2 = self.stage2_net(out_s2)
        out_s2 = out_s2.view(batch_size, self.nemb)
        
        out = self.downscale(out)
        out = self.resnet_4_0(out)
        
        out = self.downscale_v2(out)
        out = self.resnet_5_0(out)
        
        out = out.view(batch_size, 8*self.nf*self.s0*self.s0)
        out_conditioned = torch.cat((out, y_encoding), 1)
        out_s3 = self.stage3_net(actvn(out_conditioned))

        out = torch.cat((out_s1, out_s2, out_s3), 1)
        out = self.fc(actvn(out))

        return out



class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetBlock, self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fout, 3, stride=1, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        out = x_s + dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class VideoResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(VideoResnetBlock, self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv3d(self.fin, self.fout, 3, stride=1, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        out = x_s + dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class TextEncoder(nn.Module):
    def __init__(self, z_dim, inp_dim=300):
        super(TextEncoder, self).__init__()
        
        self.inp_dim = inp_dim
        self.kt=3
        self.kw=5
        self.kh=5
        self.in_filt = 32
        self.out_filt = 32

        # Filters for image discriminator

        self.content_L1 = nn.Sequential(
            nn.Conv1d(self.inp_dim, 512, 3, 1, 0),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(512, 512, 3, 1, 0),
            nn.ReLU(True),
        )
        self.cont_filter_L1 = nn.Sequential(
            nn.Linear(2560, self.in_filt*self.out_filt*self.kw*self.kh)
        )

        self.content_L2 = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(512, 256, 2, 1, 0),
            nn.ReLU(True)
        )
        self.cont_filter_L2 = nn.Sequential(
            nn.Linear(256, self.in_filt*self.out_filt*self.kw*self.kh)
        )

        # Filters for motion discriminator
        
        self.motion_L1 = nn.Sequential(
            nn.Conv1d(self.inp_dim, 512, 3, 1, 0),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(512, 512, 3, 1, 0),
            nn.ReLU(True),
        )
        self.motion_filter_L1 = nn.Sequential(
            nn.Linear(2560, self.in_filt*self.out_filt*self.kt*self.kw*self.kh)
        )

        self.motion_L2 = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(512, 256, 2, 1, 0),
            nn.ReLU(True)
        )
        self.motion_filter_L2 = nn.Sequential(
            nn.Linear(256, self.in_filt*self.out_filt*self.kt*self.kw*self.kh)
        )

    def forward(self, x):
        
        batch_size = x.size(0)

        # Content filters
        out_c_L1 = self.content_L1(x)
        c_filt_L1 = self.cont_filter_L1(out_c_L1.view(batch_size, 2560))
        c_filt_L1 = c_filt_L1.view(batch_size, self.out_filt, self.in_filt, self.kh, self.kw)

        out_c_L2 = self.content_L2(out_c_L1)
        out_c_L2 = out_c_L2.view(batch_size, 256)
        c_filt_L2 = self.cont_filter_L2(out_c_L2)
        c_filt_L2 = c_filt_L2.view(batch_size, self.out_filt, self.in_filt, self.kh, self.kw)

        c_filt = torch.cat((c_filt_L1[:, 0:self.out_filt/2, :, :, :], c_filt_L2[:, 0:self.out_filt/2, :, :, :], c_filt_L1[:, self.out_filt/2:, :, :, :], c_filt_L2[:, self.out_filt/2:, :, :, :]), dim=1)

        # Motion filters
        out_m_L1 = self.motion_L1(x)
        m_filt_L1 = self.motion_filter_L1(out_m_L1.view(batch_size, 2560))
        m_filt_L1 = m_filt_L1.view(batch_size, self.out_filt, self.in_filt, self.kt, self.kh, self.kw)

        out_m_L2 = self.motion_L2(out_m_L1)
        out_m_L2 = out_m_L2.view(batch_size, 256)
        m_filt_L2 = self.motion_filter_L2(out_m_L2)
        m_filt_L2 = m_filt_L2.view(batch_size, self.out_filt, self.in_filt, self.kt, self.kh, self.kw)

        m_filt = torch.cat((m_filt_L1[:, 0:self.out_filt/2, :, :, :], m_filt_L2[:, 0:self.out_filt/2, :, :, :], m_filt_L1[:, self.out_filt/2:, :, :, :], m_filt_L2[:, self.out_filt/2:, :, :, :]), dim=1)

        return c_filt, m_filt, out_c_L2, out_m_L2
