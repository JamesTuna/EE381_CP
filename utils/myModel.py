import torch
import torch.nn as nn
from torch.nn import functional as F

class PosEncoder(nn.Module):

    def __init__(self, embd_dim, rnn='GRU'):
        assert rnn in ('GRU','LSTM'), 'rnn must be either GRU or LSTM'
        super(PosEncoder, self).__init__()
        exec(f"self.rnn = nn.{rnn}(embd_dim,embd_dim,num_layers=1,bidirectional=True)")
        self.mlp = nn.Sequential(nn.Linear(embd_dim*2, 1024), # if not bidirecional, use embd instead of embd * 2
                                 nn.ReLU(),
                                 nn.Linear(1024, embd_dim))

    def forward(self, x):
        return x+self.mlp(self.rnn(x)[0])
    
class MyModel(nn.Module):
    def __init__(self, in_dim, out_dim, max_seq=100, embd_dim=128, 
                 rnn_type = 'GRU',n_rnn_layers=2,
                 n_transformer_layers=6, dropout=0, nheads=8, 
                 use_conv=False):
        
        super(MyModel, self).__init__()
        self.use_conv = use_conv
        
        # embedding layer
        self.embedding = nn.Linear(in_dim, embd_dim)
        
        # trainable positional encoder
        self.pos_encoder = nn.ModuleList([PosEncoder(embd_dim,rnn=rnn_type) 
                                          for i in range(n_rnn_layers)])
        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.pos_encoder_ln = nn.LayerNorm(embd_dim)
        
        # transformer layers
        transformer_layers = [nn.TransformerEncoderLayer(embd_dim, nhead=nheads, dropout=dropout) 
                              for i in range(n_transformer_layers)]
        self.transformer_layers = nn.ModuleList(transformer_layers)
        self.downsample = nn.Linear(embd_dim*2,embd_dim) 
        self.clf = nn.Linear(embd_dim, out_dim)
        
        # optional conv and deconv layers
        if self.use_conv:
            nlayers = n_transformer_layers
            self.conv_layers = nn.ModuleList([
                                                nn.Conv1d(embd_dim,embd_dim,
                                                         (nlayers-i)*2-1,stride=1,padding=0) 
                                                for i in range(nlayers)
                                                ])
            
            self.conv_ln = nn.ModuleList([nn.LayerNorm(embd_dim) for i in range(nlayers)])
            
            self.deconv_layers = nn.ModuleList([
                                                nn.ConvTranspose1d(embd_dim,embd_dim,
                                                                   (nlayers-i)*2-1,stride=1,padding=0) 
                                                for i in range(nlayers)
                                                ])
            self.deconv_ln = nn.ModuleList([nn.LayerNorm(embd_dim) for i in range(nlayers)])
        

    def forward(self, x):
        device = x.device
        x=self.embedding(x)
        x = x.permute(1, 0, 2) # (L,N,feature_dim)
        
        for pos_encoder_layer in self.pos_encoder:
            pos_encoder_layer.rnn.flatten_parameters()
            x=pos_encoder_layer(x)

        x = self.pos_encoder_dropout(x)
        x = self.pos_encoder_ln(x)
        
        if not self.use_conv:
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(x)
        else:
            enhanced_transformer_layers = zip(self.conv_layers,self.conv_ln,
                                              self.transformer_layers,
                                              self.deconv_layers,self.deconv_ln)
            for conv, convln, transformer_layer, deconv, deconvln in enhanced_transformer_layers:
                x_ = convln(F.relu(conv(x.permute(1,2,0)).permute(2,0,1)))
                x_ = transformer_layer(x_)
                x_ = deconvln(F.relu(deconv(x_.permute(1,2,0)).permute(2,0,1)))
                x += x_
                

        x = x.permute(1, 0, 2)

        output = self.clf(x)

        return output.squeeze(-1)
    
