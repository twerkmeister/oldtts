# coding: utf-8
import torch
from torch import nn
from math import sqrt
from layers.tacotron import Prenet, Encoder, Decoder, PostCBHG
from utils.generic_utils import sequence_mask


class Tacotron(nn.Module):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 linear_dim=1025,
                 mel_dim=80,
                 speaker_embedding_dim=64,
                 r=5,
                 padding_idx=None,
                 memory_size=5,
                 attn_win=False,
                 attn_norm="sigmoid"):
        super(Tacotron, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(num_chars, 256, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(0, 0.3)
        self.speaker_embedding = nn.Embedding(num_speakers,
                                              speaker_embedding_dim)
        self.encoder = Encoder(256)
        self.decoder = Decoder(256 + speaker_embedding_dim, mel_dim, r,
                               memory_size, attn_win, attn_norm)
        self.postnet = PostCBHG(mel_dim)
        self.last_linear = nn.Sequential(
            nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim),
            nn.Sigmoid())

    def forward(self, characters, text_lengths, speaker_ids, mel_specs):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)

        speaker_embeddings = self.speaker_embedding(speaker_ids)
        speaker_embeddings.unsqueeze_(-1)
        speaker_embeddings = speaker_embeddings.expand(-1, -1,
                                                       encoder_outputs.shape[1])
        speaker_embeddings = speaker_embeddings.transpose(1, 2)

        concatenated = torch.cat((encoder_outputs, speaker_embeddings), 2)

        mel_outputs, alignments, stop_tokens = self.decoder(
            concatenated, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def inference(self, characters, speaker_id):
        B = characters.size(0)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        speaker_embeddings = self.speaker_embedding(speaker_id)
        speaker_embeddings.unsqueeze_(-1)
        speaker_embeddings = speaker_embeddings.expand(-1, -1,
                                                       encoder_outputs.shape[1])
        speaker_embeddings = speaker_embeddings.transpose(1, 2)

        concatenated = torch.cat((encoder_outputs, speaker_embeddings), 2)
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            concatenated)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens
