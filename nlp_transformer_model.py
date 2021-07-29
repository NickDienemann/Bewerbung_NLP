"""
this file contains a transformer model. See following link for info: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

#imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder,TransformerEncoderLayer


class TransformerModel(nn.Module):
    """
    class that serves as a transformer 
    """

    def __init__(self,ntoken,ninp,nhead,nhid,nlayers,dropout=0.5):
        """
        task: ini the model \n
        parameters:\n
        return value:
        """

        super(TransformerModel,self).__init__()
        self.model_type ="Transformer"
        self.pos_encoder = PositionalEncoding(ninp,dropout)
        encoder_layers = TransformerEncoderLayer(ninp,nhead,nhid,dropout)
        self.transformer_encoder= TransformerEncoder(encoder_layers,nlayers)
        self.encoder = nn.Embedding(ntoken,ninp)
        self.ninp
        self.decoder = nn.Linear(ninp,ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self,sz):
        """
        task:  \n
        parameters:\n
        return value:
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        """
        task:  \n
        parameters:\n
        return value:
        """

        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src,src_mask):
        """
        task:\n
        parameters:\n
        return value:
        """
        
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output