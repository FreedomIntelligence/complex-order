# encoder.py

from torch import nn
from train_utils import clones
from sublayer import LayerNorm, SublayerOutput

class Encoder(nn.Module):
    '''
    Transformer Encoder
    
    It is a stack of N layers.
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm= LayerNorm(layer.size)
        
    def forward(self, x_real,x_phase, mask=None):
        for layer in self.layers:

            x_real,x_phase= layer(x_real,x_phase, mask)
            # x_phase = layer(, mask)

        return self.norm(x_real),self.norm(x_phase)
    
class EncoderLayer(nn.Module):
    '''
    An encoder layer
    
    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size

    def forward(self, x,y, mask=None):
        "Transformer Encoder"
        x,y= self.self_attn(x, x,x,y,y,y, mask) # Encoder self-attention
        x= self.sublayer_output[0](x, lambda x: x)
        y= self.sublayer_output[0](y, lambda x: y)

        return self.sublayer_output[1](x, self.feed_forward),self.sublayer_output[1](y, self.feed_forward)