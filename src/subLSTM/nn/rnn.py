import torch
import torch.nn as nn
import torch.jit as jit
import warnings
from torch import Tensor
from torch.nn import Parameter
from collections import namedtuple
from typing import List, Tuple, Optional
from .cell import *


class PremulLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(PremulLayer, self).__init__()

        input_size, hidden_size = cell_args
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.cell = cell(input_size, hidden_size)

    
    def forward(self,
                inputs: Tensor,
                state: Tuple[Tensor, Tensor]
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        inputs = torch.tensordot(
            inputs, self.weight_ih.t(), 
            dims=([2], [0])).unbind(0)

        outputs: List[Tensor] = []

        for i_t in inputs:
            out, state = self.cell(i_t, state)
            outputs.append(out)

        return torch.stack(outputs), state


class GRNLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(GRNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self,
                input: Tensor,
                state: Tuple[Tensor, Tensor]
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        inputs = input.unbind(0)
        outputs: List[Tensor] = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseGRNLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReverseGRNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self,
                input:Tensor,
                state:Tuple[Tensor, Tensor]
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        inputs = reverse(input.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirLayer, self).__init__()
        self.directions = nn.ModuleList([
            GRNLayer(cell, *cell_args),
            ReverseGRNLayer(cell, *cell_args),
        ])

    def forward(self,
                input: Tensor,
                states: List[Tuple[Tensor, Tensor]]
                ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:

        outputs: List[Tensor] = []
        output_states: List[Tuple[Tensor, Tensor]] = []

        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, cell, input_size, hidden_size):
    layers = [layer(cell, input_size, hidden_size)] + \
             [layer(cell, hidden_size, hidden_size) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


@jit.script
def init_states(inputs: Tensor,
                num_layers: int,
                hidden_size: int
                ) -> List[Tuple[Tensor, Tensor]]:

    states: List[Tuple[Tensor, Tensor]] = []
    temp = torch.zeros(num_layers, inputs.size(1),
                       hidden_size,2, device=inputs.device).unbind(0)

    for s in temp:
        hx, cx = s.unbind(2)
        states.append((hx, cx))

    return states


class SubLSTM(nn.Module):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ['layers', 'num_layers', 'batch_first', 'hidden_size']

    def __init__(self, input_size, hidden_size, num_layers, bias=True,
                batch_first=False, dropout=0.0, bidirectional=False,
                layer_norm=False, fixed_forget=False):
        super(SubLSTM, self).__init__()

        layer = BidirLayer if bidirectional else PremulLayer
        if fixed_forget:
            cell = LayerNormFixSubLSTMCell if layer_norm else fixSubLSTMCell
        else:
            cell = LayerNormSubLSTMCell if layer_norm else PremulSubLSTMCell

        self.layers = init_stacked_lstm(
            num_layers, layer, cell, input_size, hidden_size)

        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")

        self.dropout_layer = nn.Dropout(dropout)

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size


    def forward(self,
                input: Tensor,
                states: Optional[List[Tuple[Tensor, Tensor]]]
                ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:

        output = input if not self.batch_first else input.transpose(0, 1)
        output_states: List[Tuple[Tensor, Tensor]] = []

        if states is None:
            states = init_states(output, self.num_layers, self.hidden_size)

        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)

            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)

            output_states.append(out_state)
            i += 1

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, output_states
