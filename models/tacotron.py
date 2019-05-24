# coding: utf-8
import torch
from torch import nn
from math import sqrt

from layers.style_encoder import GlobalStyleTokens
from layers.tacotron import Prenet, Encoder, Decoder, PostCBHG
from layers.tacotron2 import Postnet
from utils.generic_utils import sequence_mask


class Tacotron(nn.Module):

    def __init__(self,
                 num_chars,
                 c,
                 padding_idx=None):
        super(Tacotron, self).__init__()
        self.r = c.r
        self.mel_dim = c.audio['num_mels']
        self.linear_dim = c.audio['num_freq']
        self.embedding = nn.Embedding(num_chars, 256, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(256)
        self.decoder = Decoder(256 + c.style_token_dim, self.mel_dim, c.r,
                               c.memory_size, c.windowing, c.attention_norm)
        self.postnet = Postnet(self.mel_dim, num_convs=5, num_feature_maps=256,
                               dropout=0.1)
        self.global_style_tokens = GlobalStyleTokens(self.mel_dim,
                                                     c.num_style_tokens,
                                                     c.style_token_dim,
                                                     c.prosody_encoding_dim,
                                                     c.scoring_func_name,
                                                     c.use_separate_keys)

        # self.postnet = PostCBHG(self.mel_dim)
        # self.last_linear = nn.Sequential(
        #     nn.Linear(self.postnet.cbhg.gru_features * 2, self.linear_dim),
        #     nn.Sigmoid())

    def shape_outputs(self, mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, characters, text_lengths, mel_specs):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        style_encoding, token_scores = self.global_style_tokens(mel_specs)
        style_encoding = style_encoding.expand(-1, encoder_outputs.size(1),
                                               -1)

        concatenated = torch.cat((encoder_outputs, style_encoding), 2)

        # encoder_outputs = encoder_outputs + style_encoding
        mel_outputs, alignments, stop_tokens = self.decoder(
            concatenated, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, \
               stop_tokens, token_scores

    def inference(self, characters, token_scores):
        B = characters.size(0)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        style_encoding = self.global_style_tokens.inference(token_scores)
        style_encoding = style_encoding.expand(-1, encoder_outputs.size(1),
                                               -1)

        concatenated = torch.cat((encoder_outputs, style_encoding), 2)

        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            concatenated)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

        # linear_outputs = self.postnet(mel_outputs)
        # linear_outputs = self.last_linear(linear_outputs)
        # return mel_outputs, linear_outputs, alignments, stop_tokens

    def get_token_scores(self, mel_specs):
        style_encoding, token_scores = self.global_style_tokens(mel_specs)
        return token_scores
