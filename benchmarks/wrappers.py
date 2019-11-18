import torch
import torch.nn as nn
import torch.jit as jit



from subLSTM.basic.nn import SubLSTM as VanillaSubLSTM
from subLSTM.torchscript.rnn import SubLSTM as ScriptedSubLSTM


class RNNClassifier(nn.Module):
    def __init__(self, rnn, rnn_output_size, n_classes):
        super(RNNClassifier, self).__init__()
        self.n_classes = n_classes
        self.rnn_output_size = rnn_output_size
        self.rnn = rnn
        self.linear = nn.Linear(rnn_output_size, n_classes)
        self.output_layer = nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        probs = self.output_layer(self.linear(output[:, -1, :]))

        return probs, hidden


class RNNRegressor(nn.Module):
    def __init__(self, rnn, rnn_output_size, responses):
        super(RNNRegressor, self).__init__()
        self.responses = responses
        self.rnn_output_size = rnn_output_size
        self.rnn = rnn
        self.linear = nn.Linear(rnn_output_size, responses)

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        predicted = self.linear(output[:, -1, :])

        return predicted, hidden


def init_model(model_type, hidden_size, input_size, n_layers,
                output_size, dropout, device, class_task=True, script=False):
    if model_type == 'subLSTM':
        if script:
            rnn = jit.script(ScriptedSubLSTM(input_size=input_size,
                                             hidden_size=hidden_size,
                                             num_layers=n_layers,
                                             fixed_forget=False,
                                             batch_first=True,
                                             dropout=dropout))

        else: 
            rnn = VanillaSubLSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=n_layers,
                                 fixed_forget=False,
                                 batch_first=True,
                                 dropout=dropout)

    elif model_type == 'fix-subLSTM':
        if script:
            rnn = jit.script(ScriptedSubLSTM(input_size=input_size,
                                             hidden_size=hidden_size,
                                             num_layers=n_layers,
                                             fixed_forget=True,
                                             batch_first=True,
                                             dropout=dropout))

        else: 
            rnn = VanillaSubLSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=n_layers,
                                 fixed_forget=True,
                                 batch_first=True,
                                 dropout=dropout)

    elif model_type == 'LSTM':
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    elif model_type == 'GRU':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    else:
        raise ValueError('Unrecognized RNN type')

    if class_task:
        model = RNNClassifier(
            rnn=rnn, rnn_output_size=hidden_size, n_classes=output_size
        ).to(device=device)
    else:
        model = RNNRegressor(
            rnn=rnn, rnn_output_size=hidden_size, responses=output_size
        ).to(device=device)

    return model
