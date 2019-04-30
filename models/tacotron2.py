from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers.tacotron2 import Encoder, Decoder, Postnet
from utils.generic_utils import sequence_mask


# TODO: match function arguments with tacotron
class Tacotron2(nn.Module):
    def __init__(self, num_chars, r, num_speakers, speaker_embedding_dim=64,
                 attn_win=False, attn_norm="softmax", prenet_type="original",
                 forward_attn=False, trans_agent=False):
        super(Tacotron2, self).__init__()
        self.n_mel_channels = 80
        self.n_frames_per_step = r
        self.embedding = nn.Embedding(num_chars, 512)
        self.speaker_embedding = nn.Embedding(num_speakers,
                                              speaker_embedding_dim)
        std = sqrt(2.0 / (num_chars + 512))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(512)
        self.decoder = Decoder(512, self.n_mel_channels, r, attn_win, attn_norm,
                               prenet_type, forward_attn, trans_agent)
        self.postnet = Postnet(self.n_mel_channels)

    def shape_outputs(self, mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text, text_lengths, speaker_ids, mel_specs=None):
        # compute mask for padding
        mask = sequence_mask(text_lengths).to(text.device)
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        speaker_embeddings = self.speaker_embedding(speaker_ids)
        speaker_embeddings.unsqueeze_(-1)
        speaker_embeddings = speaker_embeddings.expand(-1, -1,
                                                       encoder_outputs.shape[1])
        speaker_embeddings = speaker_embeddings.transpose(1, 2)

        concatenated = torch.cat((encoder_outputs, speaker_embeddings), 2)

        mel_outputs, stop_tokens, alignments = self.decoder(
            concatenated, mel_specs, mask)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

    def inference(self, text, speaker_id):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        speaker_embeddings = self.speaker_embedding(speaker_id)
        speaker_embeddings.unsqueeze_(-1)
        speaker_embeddings = speaker_embeddings.expand(-1, -1,
                                                       encoder_outputs.shape[1])
        speaker_embeddings = speaker_embeddings.transpose(1, 2)

        concatenated = torch.cat((encoder_outputs, speaker_embeddings), 2)
        mel_outputs, stop_tokens, alignments = self.decoder.inference(
            concatenated)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

    def inference_truncated(self, text, speaker_id):
        """
        Preserve model states for continuous inference
        """
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)

        speaker_embeddings = self.speaker_embedding(speaker_id)
        speaker_embeddings.unsqueeze_(-1)
        speaker_embeddings = speaker_embeddings.expand(-1, -1,
                                                       encoder_outputs.shape[1])
        speaker_embeddings = speaker_embeddings.transpose(1, 2)

        concatenated = torch.cat((encoder_outputs, speaker_embeddings), 2)

        mel_outputs, stop_tokens, alignments = self.decoder.inference_truncated(
            concatenated)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens
