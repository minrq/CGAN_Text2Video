# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd


class Trainer(object):
    def __init__(self, generator, image_discriminator, video_discriminator, text_encoder, g_optimizer, d_optimizer, vd_optimizer, enc_optimizer,
                 gan_type, reg_type, img_reg_param, vid_reg_param, batch_size, video_len=16, device=None):
        self.generator = generator
        self.image_discriminator = image_discriminator
        self.video_discriminator = video_discriminator
        self.text_encoder = text_encoder
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.vd_optimizer = vd_optimizer
        self.enc_optimizer = enc_optimizer
        self.device=device

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.img_reg_param = img_reg_param
        self.vid_reg_param = vid_reg_param
        self.batch_size = batch_size
        self.video_len = video_len

    def generator_trainstep(self, y_encoding):
        toogle_grad(self.generator, True)
        toogle_grad(self.image_discriminator, False)
        toogle_grad(self.video_discriminator, False)
        toogle_grad(self.text_encoder, False)

        self.generator.train()
        self.image_discriminator.train()
        self.video_discriminator.train()
        self.text_encoder.train()
        self.g_optimizer.zero_grad()
        # self.enc_optimizer.zero_grad()

        cont_filt, mot_filt, y_cont, y_mot = self.text_encoder(y_encoding)
        z = self.generator.sample_z_video(self.batch_size, y_cont, y_mot, video_len=self.video_len, device=self.device)
        x_fake = self.generator(z)
        d_fake_img = self.image_discriminator(x_fake, cont_filt, y_cont)
        d_fake_vid = self.video_discriminator(x_fake, mot_filt, y_mot)
        gloss_img = self.compute_loss(d_fake_img, 1)
        gloss_vid = self.compute_loss(d_fake_vid, 1)
        gloss = gloss_img + gloss_vid
        gloss.backward()

        self.g_optimizer.step()
        # self.enc_optimizer.step()

        return gloss.item()

    def image_discriminator_trainstep(self, x_real, y_encoding, y_encoding_neg):
        toogle_grad(self.generator, False)
        toogle_grad(self.image_discriminator, True)
        toogle_grad(self.video_discriminator, False)
        toogle_grad(self.text_encoder, True)

        self.generator.train()
        self.image_discriminator.train()
        self.video_discriminator.train()
        self.text_encoder.train()
        self.d_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        
        # On real data
        
        x_real.requires_grad_()
        cont_filt, mot_filt, y_cont, y_mot = self.text_encoder(y_encoding)
        cont_filt_neg, mot_filt_neg, y_cont_neg, y_mot_neg = self.text_encoder(y_encoding_neg)

        d_real = self.image_discriminator(x_real, cont_filt, y_cont)
        dloss_real = self.compute_loss(d_real, 1)
       
        if self.reg_type == 'real':
            dloss_real.backward(retain_graph=True)
            reg = self.img_reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward(retain_graph=True)
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            z = self.generator.sample_z_video(self.batch_size, y_cont, y_mot, video_len=self.video_len, device=self.device)
            x_fake = self.generator(z)

        x_fake.requires_grad_()
        
        d_real_neg = self.image_discriminator(x_real, cont_filt_neg, y_cont_neg)
        dloss_real_neg = self.compute_loss(d_real_neg, 0)

        d_fake = self.image_discriminator(x_fake, cont_filt, y_cont)
        dloss_fake_gen = self.compute_loss(d_fake, 0)

        dloss_fake = 0.5*(dloss_real_neg + dloss_fake_gen)

        if self.reg_type == 'fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.img_reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        self.d_optimizer.step()
        self.enc_optimizer.step()
        
        toogle_grad(self.image_discriminator, False)
        toogle_grad(self.text_encoder, False)
        
        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def video_discriminator_trainstep(self, x_real, y_encoding, y_encoding_neg):
        toogle_grad(self.generator, False)
        toogle_grad(self.image_discriminator, False)
        toogle_grad(self.video_discriminator, True)
        toogle_grad(self.text_encoder, True)
        self.generator.train()
        self.image_discriminator.train()
        self.video_discriminator.train()
        self.text_encoder.train()
        self.vd_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        
        # On real data
        
        x_real.requires_grad_()
        cont_filt, mot_filt, y_cont, y_mot = self.text_encoder(y_encoding)
        cont_filt_neg, mot_filt_neg, y_cont_neg, y_mot_neg = self.text_encoder(y_encoding_neg)
        
        d_real = self.video_discriminator(x_real, mot_filt, y_mot)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real':
            dloss_real.backward(retain_graph=True)
            reg = self.vid_reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward(retain_graph=True)
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            z = self.generator.sample_z_video(self.batch_size, y_cont, y_mot, video_len=self.video_len, device=self.device)
            x_fake = self.generator(z)

        x_fake.requires_grad_()
        d_fake = self.video_discriminator(x_fake, mot_filt, y_mot)
        dloss_fake_gen = self.compute_loss(d_fake, 0)

        d_real_neg = self.video_discriminator(x_real, mot_filt_neg, y_mot_neg)
        dloss_real_neg = self.compute_loss(d_real_neg, 0)

        dloss_fake = 0.5*(dloss_fake_gen + dloss_real_neg)

        if self.reg_type == 'fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.vid_reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        self.vd_optimizer.step()
        self.enc_optimizer.step()
        
        toogle_grad(self.video_discriminator, False)
        toogle_grad(self.text_encoder, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()


    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - 1.).pow(2).mean()

        return reg


# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
