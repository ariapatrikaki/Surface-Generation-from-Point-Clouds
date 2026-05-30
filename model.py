import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # 8 fully-connected layers total:
        # 3 -> 512 -> 512 -> 512 -> 509 -> concat input(3) -> 512 -> 512 -> 512 -> 1
        # Weight norm + PReLU(shared slope) + dropout on the first 7 FC layers.
        self.dropout_prob = dropout_prob
        self.use_tanh = use_tanh
        self.dropout_layers = set(dropout) if dropout is not None else set()
        self.norm_layers = set(norm_layers)
        self.latent_in = set(latent_in)

        in_dims =  [3,   512, 512, 512, 512, 512, 512, 512]
        out_dims = [512, 512, 512, 509, 512, 512, 512,   1]
        self.num_layers = len(in_dims)

        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            in_dim = in_dims[layer_idx]
            out_dim = out_dims[layer_idx]

            layer = nn.Linear(in_dim, out_dim)
            if weight_norm and (layer_idx in self.norm_layers) and (layer_idx != self.num_layers - 1):
                layer = nn.utils.weight_norm(layer)
            self.layers.append(layer)

        # Common learnable negative slope for all channels
        self.activation = nn.PReLU(num_parameters=1)
        self.th = nn.Tanh()
        # ***********************************************************************
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        x = input
        xyz = input

        for layer_idx, layer in enumerate(self.layers):
            # After the 4th FC layer (0-based index 3), concatenate xyz before the next FC.
            if layer_idx in self.latent_in:
                x = torch.cat([x, xyz], dim=1)

            x = layer(x)

            # First seven FC layers use activation + dropout.
            if layer_idx != self.num_layers - 1:
                x = self.activation(x)
                if layer_idx in self.dropout_layers:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.use_tanh:
            x = self.th(x)
        # ***********************************************************************
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################

        return x
