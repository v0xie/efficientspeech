
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.tools import pad

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        
        #print("feature shape", x.shape)
        #print("duration shape", duration.shape)

        #a = x
        #b = duration.squeeze().long()
        #c = a.repeat_interleave(b, dim=1)
        #print("c shape", c.shape)

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        #stack = torch.stack(output, 0)
        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        #print("stack", stack.shape)
        #print("output shape", output.shape)
        #print("mel_len shape", mel_len)
        
        return output, torch.LongTensor(mel_len)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item() if i < predicted.shape[0] else 0
            expand_size = 1 if math.isnan(expand_size) else expand_size 
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class LengthRegulatorONNX(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item() if i < predicted.shape[0] else 0
            expand_size = 1 if math.isnan(expand_size) else expand_size 
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len