import torch

from tnpul.torchmps.tdvp.tdvp import TDVP, TDVP_V2
from tnpul.torchmps.torchmps import TI_MPS, MPS
from tnpul.torchmps.embeddings import image_embedding, angle_decoder, angle_encoder, linear_decoder, linear_encoder


class MultiTDVP(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MultiTDVP, self).__init__()
        self.kwargs = kwargs

        self.ti = kwargs["ti"]
        # if hasattr(kwargs, "ti") and kwargs["ti"]:
        #     self.ti = True
        if hasattr(kwargs, "ti_tdvp") and kwargs["ti_tdvp"]:
            self.ti = True

        self.ntdvp = kwargs["ntdvp"]
        self.nrep = kwargs["nrep"]

        self.mask_off = kwargs["mask_off"]
        self.residual = kwargs["residual"]
        self.ka = kwargs['ka']
        self.remove_trace = kwargs['remove_trace']
        self.scale = kwargs["scale"]
        self.cyclic = kwargs["cyclic"]

        # we enable the global residual if ntdvp >1.
        # If ntdvp ==1 the local residual is the same as the global one
        if self.residual and self.ntdvp*self.nrep > 1:
            self.register_parameter(name='global_res_coef',
                                    param=torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))

        tdvps = []
        for i in range(kwargs["ntdvp"], 0, -1):
            tdvp = TDVP(kwargs["input_dim"], d=kwargs["feature_dim"], D=kwargs["Dtdvp"], out_norm=kwargs["out_norm"],
                        activation=kwargs["activation"], mode=kwargs["mode"], eps=kwargs[
                            "eps_tdvp"], ti=self.ti, mask_off=self.mask_off, residual=self.residual, ka=self.ka,
                        remove_trace=self.remove_trace, cyclic=self.cyclic, scale=self.scale, trainable_boundary=kwargs['trainable_boundary'], cinit=kwargs["cinit"])
            if kwargs['cuda']:
                tdvp.to('cuda')
            tdvps.append(tdvp)

        self.tdvps = torch.nn.ModuleList(tdvps)

    def forward(self, inputs):
        x = inputs
        for tdvp in self.tdvps:
            for _ in range(self.kwargs["nrep"]):
                x = tdvp(x)

        if self.residual and self.ntdvp*self.nrep > 1:
            x = x + inputs*self.global_res_coef
            x = torch.divide(x, torch.linalg.norm(x, dim=2, keepdim=True))

        return x


class MultiTDVP_V2(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MultiTDVP_V2, self).__init__()
        self.kwargs = kwargs

        self.ti = kwargs["ti"]
        # if hasattr(kwargs, "ti") and kwargs["ti"]:
        #     self.ti = True
        if hasattr(kwargs, "ti_tdvp") and kwargs["ti_tdvp"]:
            self.ti = True

        self.ntdvp = kwargs["ntdvp"]
        self.nrep = kwargs["nrep"]

        self.residual = kwargs["residual"]
        self.scale = kwargs["scale"]

        self.din = kwargs["feature_dim"]
        self.dout = kwargs["dout"]

        # we enable the global residual if ntdvp >1.
        # If ntdvp ==1 the local residual is the same as the global one
        if self.residual and self.ntdvp*self.nrep > 1:
            self.register_parameter(name='global_res_coef',
                                    param=torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))

        tdvps = []
        for i in range(kwargs["ntdvp"]):
            if i < kwargs["ntdvp"]-1:
                dout = self.din
            else:
                dout = self.dout
            tdvp = TDVP_V2(kwargs["input_dim"], din=self.din, dout=dout, D=kwargs["Dtdvp"], out_norm=kwargs["out_norm"],
                           activation=kwargs["activation"], mode=kwargs["mode"], eps=kwargs[
                               "eps_tdvp"], ti=self.ti, cyclic=kwargs["cyclic"], residual=self.residual,
                           scale=self.scale, trainable_A=kwargs['trainable_A'])
            if kwargs['cuda']:
                tdvp.to('cuda')
            tdvps.append(tdvp)

        self.tdvps = torch.nn.ModuleList(tdvps)

    def forward(self, inputs):
        x = inputs
        for tdvp in self.tdvps:
            for _ in range(self.kwargs["nrep"]):
                x, Als, Ars = tdvp(x)

        if self.residual and self.ntdvp*self.nrep > 1 and self.din == self.dout:
            x = x + inputs*self.global_res_coef

        return x, Als, Ars


class MPS_TDVP(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MPS_TDVP, self).__init__()

        self.kwargs = kwargs
        self.ti = kwargs["ti"]
        self.ti_mps = kwargs["ti"] or kwargs["ti_mps"]

        self.multi_tdvp = MultiTDVP(**kwargs)

        self.inds = range(kwargs["input_dim"])
        if self.ti_mps:
            self.mps = TI_MPS(output_dim=kwargs['nclass'], bond_dim=kwargs['D'], feature_dim=kwargs['feature_dim'],
                              parallel_eval=False, fixed_ends=True, init_std=1e-9, use_bias=True, fixed_bias=True,
                              permute=self.inds, config=kwargs)
        else:
            self.mps = MPS(input_dim=kwargs['input_dim'], output_dim=kwargs['nclass'], bond_dim=kwargs['D'],
                           adaptive_mode=False, periodic_bc=self.ti_mps, feature_dim=kwargs['feature_dim'],
                           permute=self.inds, config=kwargs)

        if kwargs['cuda']:
            self.multi_tdvp.to('cuda')
            self.mps.to('cuda')

    def forward(self, inputs):
        if hasattr(self, "multi_tdvp"):  # New format
            x = self.multi_tdvp(inputs)
        else:  # For compatibility with the old format
            x = inputs
            for tdvp in self.tdvps:
                for _ in range(self.kwargs["nrep"]):
                    x = tdvp(x)

            if self.kwargs["ntdvp"] > 0:
                for _ in range(self.kwargs["nrep_final"]):
                    x = self.tdvp_L1(x)
        x = self.mps(x)
        return x

# TODO: Add a loss that forces the output values to be as far away from 0.5 as possible. At each step not just the final result!!!


class CA_TDVP(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CA_TDVP, self).__init__()
        self.kwargs = kwargs
        self.multi_tdvp = MultiTDVP(**kwargs)
        # self.encoder = angle_encoder
        # self.decoder = angle_decoder
        self.encoder = linear_encoder
        self.decoder = linear_decoder

        if kwargs['cuda']:
            self.multi_tdvp.to('cuda')

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.multi_tdvp(x)
        nrm = torch.linalg.norm(x, dim=-1)
        dx = self.decoder(x)
        return dx, nrm, x


class CA_TDVP_V2(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CA_TDVP_V2, self).__init__()
        self.kwargs = kwargs
        self.multi_tdvp = MultiTDVP_V2(**kwargs)
        # self.encoder = angle_encoder
        # self.decoder = angle_decoder
        self.encoder = linear_encoder
        self.decoder = linear_decoder
        self.dout = kwargs["dout"]

        if kwargs['cuda']:
            self.multi_tdvp.to('cuda')

    def forward(self, inputs):
        x = self.encoder(inputs)
        x, Als, Ars = self.multi_tdvp(x)
        if self.dout > 1:
            dx = self.decoder(x)
        else:
            dx = x[..., 0]
        return dx, x, Als, Ars
