import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from layers.tacotron2 import ConvBNBlock


class Encoder(nn.Module):
    def __init__(self, in_features=128):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(5):
            convolutions.append(
                ConvBNBlock(in_features, in_features, 5, 'relu', 0.0))
        self.convolutions = nn.Sequential(*convolutions)

    def forward(self, x):
        x = self.convolutions(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

    def inference(self, x):
        x = self.convolutions(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x


class EmbeddingCombiner(nn.Module):
    """Combining the encoder, speaker and style embedding."""
    def __init__(self, in_features, out_features=[256, 128],
                 res_encoder=True, activation="relu", dropout=0.1):
        super().__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_features, out_features)
        ])
        self.res_encoder = res_encoder
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            out_features[-1],
            out_features[-1],
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        # self.dropout = nn.Dropout(0.5)
        # self.init_layers()

    def init_layers(self):
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(
                layer.weight, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, encoder_outputs, speaker_embeddings, style_encoding,
                input_lengths):
        speaker_embeddings.unsqueeze_(1)
        speaker_embeddings = speaker_embeddings.expand(encoder_outputs.size(0),
                                                       encoder_outputs.size(1),
                                                       -1)
        style_encoding = style_encoding.expand(-1, encoder_outputs.size(1), -1)
        x = torch.cat((encoder_outputs, speaker_embeddings,
                           style_encoding), 2)
        for linear in self.layers:
            x = self.dropout(self.activation(linear(x)))

        if self.res_encoder:
            x = x + encoder_outputs

        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=True,
        )
        return outputs

    def inference(self, encoder_outputs, speaker_embeddings, style_encoding):
        speaker_embeddings.unsqueeze_(1)
        speaker_embeddings = speaker_embeddings.expand(encoder_outputs.size(0),
                                                       encoder_outputs.size(1),
                                                       -1)
        style_encoding = style_encoding.expand(-1, encoder_outputs.size(1), -1)
        x = torch.cat((encoder_outputs, speaker_embeddings,
                           style_encoding), 2)
        for linear in self.layers:
            x = self.dropout(self.activation(linear(x)))

        if self.res_encoder:
            x = x + encoder_outputs

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs
