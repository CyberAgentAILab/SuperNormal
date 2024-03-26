import torch
import torch.nn as nn
import numpy as np
import tinycudann as tcnn
from icecream import ic

class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 encoding_config=None,
                 input_concat=False):
        super(SDFNetwork, self).__init__()
        self.input_concat = input_concat

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        if encoding_config is not None:
            self.encoding = tcnn.Encoding(d_in, encoding_config).to(torch.float32)
            dims[0] = self.encoding.n_output_dims
            if input_concat:
                dims[0] += d_in
        else:
            self.encoding = None

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.bindwidth = 0
        self.enc_dim = self.encoding.n_output_dims

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif self.encoding is not None and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.encoding is not None and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)
        # self.activation = nn.ReLU()

    def increase_bandwidth(self):
        self.bindwidth += 1

    def forward(self, inputs):
        if self.encoding is not None:
            encoded = self.encoding(inputs).to(torch.float32)

            # set the dimension of the encoding to 0 if the input is outside the bandwidth
            enc_mask = torch.ones(self.enc_dim, dtype=torch.bool, device=encoded.device, requires_grad=False)
            enc_mask[self.bindwidth*2:] = 0
            encoded = encoded * enc_mask

        if self.input_concat:
            inputs = torch.cat([inputs, encoded], dim=1)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    @torch.enable_grad()
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    @torch.enable_grad()
    def divergence(self, y, x):
        div = 0.
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
        return div

    @torch.enable_grad()
    def laplace(self, x):
        return self.divergence(self.gradient(x), x)


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)