import torch
from gan_training.metrics import inception_score
from gan_training import utils
import pdb

class Evaluator(object):
    def __init__(self, generator, train_loader, glove_dict, batch_size=64, video_len=16,
                 inception_nsamples=60000, device=None):
        self.generator = generator
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.video_len = video_len
        self.train_loader = train_loader
        self.glove_dict = glove_dict
    
    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = self.generator.sample_z_video(self.batch_size, video_len=self.video_len, device=self.device)

            samples = self.generator(ztest)
            samples = samples.permute(0, 2, 1, 3, 4)
            samples = samples.view(samples.size(0)*samples.size(1), samples.size(2), samples.size(3), samples.size(4))
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def create_samples(self, text_encoder, batch_size=2, video_len=16):
        self.generator.eval()
        __, y, __, y_labels = iter(self.train_loader).next()
        y = y[0:batch_size, :]
        y = y.long()
        y_embedding = utils.get_glove_embedding(self.glove_dict, y)
        y_embedding = y_embedding.to(self.device)
        __, __, y_cont, y_mot = text_encoder(y_embedding)
        batch_size = min(batch_size, y_cont.size(0))
        ztest = self.generator.sample_z_video(batch_size, y_cont, y_mot, video_len=video_len, device=self.device)
                
        # Sample x
        with torch.no_grad():
            x = self.generator(ztest)
        return x, y, y_labels
