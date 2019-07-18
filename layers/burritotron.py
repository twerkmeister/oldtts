import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from layers.tacotron2 import ConvBNBlock


class Prenet(nn.Module):
    r""" Prenet as explained at https://arxiv.org/abs/1703.10135.
    It creates as many layers as given by 'out_features'

    Args:
        in_features (int): size of the input vector
        out_features (int or list): size of each output sample.
            If it is a list, for each value, there is created a new layer.
    """

    def __init__(self, in_features, out_features=[256, 128], dropout=0.25):
        super(Prenet, self).__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_features, out_features)
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class StopNet(nn.Module):
    r"""
    Args:
        in_features (int): feature dimension of input.
    """

    def __init__(self, in_features):
        super(StopNet, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features, 1)
        torch.nn.init.xavier_uniform_(
            self.linear.weight, gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, inputs):
        outputs = self.dropout(inputs)
        outputs = self.linear(outputs)
        return outputs


class Encoder(nn.Module):
    def __init__(self, in_features=128):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(5):
            convolutions.append(
                ConvBNBlock(in_features, in_features, 5, 'relu', 0.2))
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

    def __init__(self, num_features=128, dropout=0.1):
        super().__init__()
        in_features = [num_features*2, num_features, num_features]
        out_features = [num_features, num_features, num_features]
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_features, out_features)
        ])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            num_features,
            num_features,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

    def forward(self, encoder_outputs, speaker_embeddings, style_encoding,
                input_lengths):
        speaker_embeddings.unsqueeze_(1)
        style = torch.cat((speaker_embeddings, style_encoding), 2)
        for linear in self.layers:
            style = self.dropout(self.activation(linear(style)))

        style = style.expand(-1, encoder_outputs.size(1), -1)
        x = encoder_outputs + style

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
        style = torch.cat((speaker_embeddings, style_encoding), 2)
        for linear in self.layers:
            style = self.dropout(self.activation(linear(style)))

        style = style.expand(-1, encoder_outputs.size(1), -1)
        x = encoder_outputs + style

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    r"""Decoder module.

    Args:
        in_features (int): input vector (encoder output) sample size.
        memory_dim (int): memory vector (prev. time-step output) sample size.
        r (int): number of outputs per time step.
        memory_size (int): size of the past window. if <= 0 memory_size = r
    """

    def __init__(self, in_features, memory_dim, r, memory_size,
                 attn_windowing, attn_norm):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.max_decoder_steps = 500
        self.memory_size = memory_size if memory_size > 0 else r
        self.memory_dim = memory_dim
        # memory -> |Prenet| -> processed_memory
        self.prenet = Prenet(
            memory_dim * self.memory_size, out_features=[256, 128])
        # processed_inputs, processed_memory -> |Attention| -> Attention,
        # attention, RNN_State
        self.attention_rnn = AttentionRNNCell(
            out_dim=128,
            rnn_dim=256,
            annot_dim=in_features,
            memory_dim=128,
            align_model='ls',
            windowing=attn_windowing,
            norm=attn_norm)
        # (processed_memory | attention context) -> |Linear| ->
        # decoder_RNN_input
        self.project_to_decoder_in = nn.Linear(256 + in_features, 256)
        # decoder_RNN_input -> |RNN| -> RNN_state
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])
        # RNN_state -> |Linear| -> mel_spec
        self.proj_to_mel = nn.Linear(256, memory_dim * r)
        # learn init values instead of zero init.
        self.attention_rnn_init = nn.Embedding(1, 256)
        self.memory_init = nn.Embedding(1, self.memory_size * memory_dim)
        self.decoder_rnn_inits = nn.Embedding(2, 256)
        self.stopnet = StopNet(256 + memory_dim * r)
        # self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.project_to_decoder_in.weight,
            gain=torch.nn.init.calculate_gain('linear'))
        torch.nn.init.xavier_uniform_(
            self.proj_to_mel.weight,
            gain=torch.nn.init.calculate_gain('linear'))

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        B = memory.shape[0]
        # Grouping multiple frames if necessary
        if memory.size(-1) == self.memory_dim:
            memory = memory.contiguous()
            memory = memory.view(B, memory.size(1) // self.r, -1)
        # Time first (T_decoder, B, memory_dim)
        memory = memory.transpose(0, 1)
        return memory

    def _init_states(self, inputs):
        """
        Initialization of decoder states
        """
        B = inputs.size(0)
        T = inputs.size(1)
        # go frame as zeros matrix
        self.memory_input = self.memory_init(inputs.data.new_zeros(B).long())

        # decoder states
        self.attention_rnn_hidden = self.attention_rnn_init(
            inputs.data.new_zeros(B).long())
        self.decoder_rnn_hiddens = [
            self.decoder_rnn_inits(inputs.data.new_tensor([idx] * B).long())
            for idx in range(len(self.decoder_rnns))
        ]
        self.current_context_vec = inputs[:, 0, :].clone()
        # self.current_context_vec = inputs.data.new(B, self.in_features).zero_()
        # attention states
        self.attention = inputs.data.new(B, T).zero_()
        self.attention[:, 0] = 0.1
        self.attention_history = [inputs.data.new(B, T).zero_()] * 5
        self.delta_attention = inputs.data.new(B, T).zero_()
        self.attention_cum = inputs.data.new(B, T).zero_()
        self.attention_cum[:, 0] = 0.1

    def _parse_outputs(self, outputs, attentions, stop_tokens):
        # Back to batch first
        attentions = torch.stack(attentions).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1).squeeze(-1)
        return outputs, attentions, stop_tokens

    def decode(self,
               inputs,
               t,
               mask=None):
        # Prenet
        processed_memory = self.prenet(self.memory_input)
        # Attention RNN
        attention_delta = self.attention - self.attention_history[-1]
        attention_delta2 = self.attention - self.attention_history[-2]
        attention_delta3 = self.attention - self.attention_history[-3]
        attention_delta4 = self.attention - self.attention_history[-4]
        attention_delta5 = self.attention - self.attention_history[-5]
        attention_prev_delta = self.attention_history[-1] - \
                               self.attention_history[-2]
        attention_delta_delta = attention_delta - attention_prev_delta
        attention_prev_delta2 = self.attention_history[-1] - \
                                self.attention_history[-3]
        attention_delta_delta2 = attention_delta2 - attention_prev_delta2
        attention_prev_delta3 = self.attention_history[-1] - \
                                self.attention_history[-4]
        attention_delta_delta3 = attention_delta3 - attention_prev_delta3
        attention_prev_delta4 = self.attention_history[-1] - \
                                self.attention_history[-5]
        attention_delta_delta4 = attention_delta4 - attention_prev_delta4
        self.attention_history.append(self.attention.clone())
        attention_cat = torch.cat(
            (self.attention.unsqueeze(1),
             attention_delta.unsqueeze(1),
             attention_delta2.unsqueeze(1),
             attention_delta3.unsqueeze(1),
             attention_delta4.unsqueeze(1),
             attention_delta5.unsqueeze(1),
             attention_delta_delta.unsqueeze(1),
             attention_delta_delta2.unsqueeze(1),
             attention_delta_delta3.unsqueeze(1),
             attention_delta_delta4.unsqueeze(1),
             self.attention_cum.unsqueeze(1)), dim=1)
        # attention_cat = torch.cat(
        #     (self.attention.unsqueeze(1),
        #      self.attention_cum.unsqueeze(1)), dim=1)
        self.attention_rnn_hidden, self.current_context_vec, self.attention =\
            self.attention_rnn(
            processed_memory, self.current_context_vec,
            self.attention_rnn_hidden,
            inputs, attention_cat, mask, t)
        del attention_cat
        self.attention_cum += self.attention
        # Concat RNN output and attention context vector
        decoder_input = self.project_to_decoder_in(
            torch.cat((self.attention_rnn_hidden, self.current_context_vec),
                      -1))
        # Pass through the decoder RNNs
        for idx in range(len(self.decoder_rnns)):
            self.decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                decoder_input, self.decoder_rnn_hiddens[idx])
            # Residual connection
            decoder_input = self.decoder_rnn_hiddens[idx] + decoder_input
        decoder_output = decoder_input
        del decoder_input
        # predict mel vectors from decoder vectors
        output = self.proj_to_mel(decoder_output)
        output = torch.sigmoid(output)
        # predict stop token
        stopnet_input = torch.cat([decoder_output, output], -1)
        del decoder_output
        stop_token = self.stopnet(stopnet_input)
        return output, stop_token, self.attention

    def _update_memory_queue(self, new_memory):
        if self.memory_size > 0:
            self.memory_input = torch.cat([
                self.memory_input[:, self.r * self.memory_dim:].clone(),
                new_memory
            ], dim=-1)
        else:
            self.memory_input = new_memory

    def forward(self, inputs, memory, mask):
        """
        Args:
            inputs: Encoder outputs.
            memory: Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses
              the last
              output as the input.
            mask: Attention mask for sequence padding.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - memory: batch x #mel_specs x mel_spec_dim
        """
        # Run greedy decoding if memory is None
        memory = self._reshape_memory(memory)
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        while len(outputs) < memory.size(0):
            if t > 0:
                new_memory = memory[t - 1]
                self._update_memory_queue(new_memory)
            output, stop_token, attention = self.decode(inputs, t, mask)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1

        return self._parse_outputs(outputs, attentions, stop_tokens)

    def inference(self, inputs):
        """
        Args:
            inputs: Encoder outputs.

        Shapes:
            - inputs: batch x time x encoder_out_dim
        """
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        while True:
            if t > 0:
                new_memory = outputs[-1]
                self._update_memory_queue(new_memory)
            output, stop_token, attention = self.decode(inputs, t, None)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1
            if t > inputs.shape[1] / 4 and (stop_token > 0.6
                                            or attention[:, -1].item() > 0.6):
                break
            elif t > self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break
        return self._parse_outputs(outputs, attentions, stop_tokens)


class BahdanauAttention(nn.Module):
    def __init__(self, annot_dim, query_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, annots, query):
        """
        Shapes:
            - annots: (batch, max_time, dim)
            - query: (batch, 1, dim) or (batch, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)
        # (batch, max_time, 1)
        alignment = self.v(torch.tanh(processed_query + processed_annots))
        # (batch, max_time)
        return alignment.squeeze(-1)


class LocationSensitiveAttention(nn.Module):
    """Location sensitive attention following
    https://arxiv.org/pdf/1506.07503.pdf"""

    def __init__(self,
                 annot_dim,
                 query_dim,
                 attn_dim,
                 kernel_size=15,
                 filters=128):
        super(LocationSensitiveAttention, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        padding = [(kernel_size - 1) // 2, (kernel_size - 1) // 2]
        self.loc_conv = nn.Sequential(
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(
                11,
                filters,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=False))
        self.loc_linear = nn.Linear(filters, attn_dim, bias=True)
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Sequential(nn.Linear(attn_dim, 64, bias=False),
                               nn.Linear(64, 1, bias=False))
        self.processed_annots = None
        # self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.loc_linear.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(
            self.query_layer.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(
            self.annot_layer.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(
            self.v.weight,
            gain=torch.nn.init.calculate_gain('linear'))

    def reset(self):
        self.processed_annots = None

    def forward(self, annot, query, loc):
        """
        Shapes:
            - annot: (batch, max_time, dim)
            - query: (batch, 1, dim) or (batch, dim)
            - loc: (batch, 2, max_time)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        processed_loc = self.loc_linear(self.loc_conv(loc).transpose(1, 2))
        processed_query = self.query_layer(query)
        # cache annots
        if self.processed_annots is None:
            self.processed_annots = self.annot_layer(annot)
        alignment = self.v(
            torch.tanh(processed_query + self.processed_annots + processed_loc))
        del processed_loc
        del processed_query
        # (batch, max_time)
        return alignment.squeeze(-1)


class AttentionRNNCell(nn.Module):
    def __init__(self, out_dim, rnn_dim, annot_dim, memory_dim, align_model,
                 windowing=False, norm="sigmoid"):
        r"""
        General Attention RNN wrapper

        Args:
            out_dim (int): context vector feature dimension.
            rnn_dim (int): rnn hidden state dimension.
            annot_dim (int): annotation vector feature dimension.
            memory_dim (int): memory vector (decoder output) feature dimension.
            align_model (str): 'b' for Bahdanau, 'ls' Location Sensitive
            alignment.
            windowing (bool): attention windowing forcing monotonic attention.
                It is only active in eval mode.
            norm (str): norm method to compute alignment weights.
        """
        super().__init__()
        self.align_model = align_model
        self.rnn_cell = nn.GRUCell(annot_dim + memory_dim, rnn_dim)
        self.windowing = windowing
        if self.windowing:
            self.win_back = 3
            self.win_front = 6
            self.win_idx = None
        self.norm = norm
        if align_model == 'b':
            self.alignment_model = BahdanauAttention(annot_dim, rnn_dim,
                                                     out_dim)
        if align_model == 'ls':
            self.alignment_model = LocationSensitiveAttention(
                annot_dim, rnn_dim, out_dim)
        else:
            raise RuntimeError(" Wrong alignment model name: {}. Use\
                'b' (Bahdanau) or 'ls' (Location Sensitive).".format(
                align_model))

    def forward(self, memory, context, rnn_state, annots, atten, mask, t):
        """
        Shapes:
            - memory: (batch, 1, dim) or (batch, dim)
            - context: (batch, dim)
            - rnn_state: (batch, out_dim)
            - annots: (batch, max_time, annot_dim)
            - atten: (batch, 2, max_time)
            - mask: (batch,)
        """
        if t == 0:
            self.alignment_model.reset()
            self.win_idx = 0
        rnn_output = self.rnn_cell(torch.cat((memory, context), -1), rnn_state)
        if self.align_model is 'b':
            alignment = self.alignment_model(annots, rnn_output)
        else:
            alignment = self.alignment_model(annots, rnn_output, atten)
        if mask is not None:
            mask = mask.view(memory.size(0), -1)
            alignment.masked_fill_(1 - mask, -float("inf"))
        # Windowing
        if not self.training and self.windowing:
            back_win = self.win_idx - self.win_back
            front_win = self.win_idx + self.win_front
            if back_win > 0:
                alignment[:, :back_win] = -float("inf")
            if front_win < memory.shape[1]:
                alignment[:, front_win:] = -float("inf")
            # Update the window
            self.win_idx = torch.argmax(alignment, 1).long()[0].item()
        if self.norm == "softmax":
            alignment = torch.softmax(alignment, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(alignment) / torch.sigmoid(alignment).sum(
                dim=1).unsqueeze(1)
        else:
            raise RuntimeError("Unknown value for attention norm type")
        context = torch.bmm(alignment.unsqueeze(1), annots)
        context = context.squeeze(1)
        return rnn_output, context, alignment
