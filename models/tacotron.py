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
                 linear_dim=1025,
                 mel_dim=80,
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
        self.encoder = Encoder(256)
        self.decoder = Decoder(256 + 64 + 4, mel_dim, r, memory_size, attn_win,
                               attn_norm)
        self.postnet = Postnet(mel_dim, num_convs=5, num_feature_maps=256,
                               dropout=0.1)
        self.global_style_tokens = GlobalStyleTokens(mel_dim, 128,
                                                     64, 128)

    def shape_outputs(self, mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, characters, text_lengths, mel_specs, timers):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        style_encoding = self.global_style_tokens(mel_specs)
        style_encoding = style_encoding.expand(-1, encoder_outputs.size(1),
                                               -1)

        concatenated = torch.cat((encoder_outputs, style_encoding, timers), 2)

        # encoder_outputs = encoder_outputs + style_encoding
        mel_outputs, alignments, stop_tokens = self.decoder(
            concatenated, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

    def inference(self, characters):
        B = characters.size(0)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        style_encoding = torch.zeros((1, 1, 64)).cuda(non_blocking=True)
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
