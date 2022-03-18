import torch
from gan_training.metrics import inception_score
from gan_training import utils
import numpy as np

class Evaluator(object):
    def __init__(self, generator, train_loader, vocab, glove_dict, batch_size=64, video_len=16,
                 inception_nsamples=60000, device=None):
        self.generator = generator
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.video_len = video_len
        self.train_loader = train_loader
        self.glove_dict = glove_dict
        self.vocab = vocab
    
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

    def text_mapper(self, sentence):
        
        encoding = []
        
        sentence = sentence.split(' ')
        for w in sentence:
            w = w.lower()
            encoding.append(self.vocab[w])

        encoding = np.array(encoding)
        encoding_vec = np.zeros((16))
        encoding_vec[0:len(encoding)] = encoding
        encoding_vec = torch.from_numpy(encoding_vec)

        return encoding_vec

    def create_samples(self, text_encoder, text_list, batch_size=2, video_len=16):
        self.generator.eval()
        
        assert len(text_list) == 2
        
        # Get encodings
        
        enc1 = self.text_mapper(text_list[0])
        enc2 = self.text_mapper(text_list[1])
        enc1 = enc1.long()
        enc2 = enc2.long()

        enc1 = enc1.unsqueeze(0)
        enc2 = enc2.unsqueeze(0)

        enc1_embedding = utils.get_glove_embedding(self.glove_dict, enc1)
        enc1_embedding = enc1_embedding.to(self.device)
        enc2_embedding = utils.get_glove_embedding(self.glove_dict, enc2)
        enc2_embedding = enc2_embedding.to(self.device)

        __, __, y_cont1, y_mot1 = text_encoder(enc1_embedding)
        __, __, y_cont2, y_mot2 = text_encoder(enc2_embedding)
        ztest = self.generator.sample_z_video(batch_size, y_cont1, y_mot1, y_cont2, y_mot2, video_len=video_len, device=self.device)

        
        # Sample x
        with torch.no_grad():
            x = self.generator(ztest)
        return x, enc1, enc2
