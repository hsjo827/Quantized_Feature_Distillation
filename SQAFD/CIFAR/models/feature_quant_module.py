import torch
import torch.nn as nn
import math
from .custom_modules import STE_discretizer, EWGS_discretizer
import logging

class FeatureQuantizer(nn.Module):
    def __init__(self, num_levels, scaling_factor, baseline=False, use_student_quant_params=False):
        super(FeatureQuantizer, self).__init__()
        self.num_levels = num_levels
        self.scaling_factor = scaling_factor
        self.baseline = baseline
        self.use_student_quant_params = use_student_quant_params

        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply

        self.uF = nn.Parameter(torch.tensor(1.0), requires_grad=not use_student_quant_params)
        self.lF = nn.Parameter(torch.tensor(0.0), requires_grad=not use_student_quant_params)
        self.register_buffer('bkwd_scaling_factorF', torch.tensor(scaling_factor).float())

        self.hook_feature_values = False
        self.buff_feature = None

        self.register_buffer('init', torch.tensor([0]))
        self.output_scale = nn.Parameter(torch.tensor(1.0), requires_grad=not use_student_quant_params)


    def initialize(self, x):
        self.uF.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
        self.lF.data.fill_(x.min())

        x = (x - self.lF) / (self.uF - self.lF)
        x = x.clamp(min=0, max=1)

        if not self.baseline:
            x_quant = self.EWGS_discretizer(x, self.num_levels, self.bkwd_scaling_factorF)
        else:
            x_quant = self.STE_discretizer(x, self.num_levels)

        x_abs_mean = x.abs().mean()
        x_quant_abs_mean = x_quant.abs().mean()
        self.output_scale.data.fill_(x_abs_mean / x_quant_abs_mean)

        self.init.fill_(0)


    def forward(self, x, save_dict=None, quant_params=None):
        if self.init == 1:
            self.initialize(x)

        if self.use_student_quant_params and quant_params is not None:
            old_lF = self.lF.item()
            old_uF = self.uF.item()
            old_scale = self.output_scale.item()

            self.lF.data.fill_(quant_params.get("lA", self.lF.item()))
            self.uF.data.fill_(quant_params.get("uA", self.uF.item()))
            self.output_scale.data.fill_(quant_params.get("output_scale", self.output_scale.item()))
            
            logging.info(f"Feature quantizer lF updated: {old_lF:.4f} -> {self.lF.item():.4f}")
            logging.info(f"Feature quantizer uF updated: {old_uF:.4f} -> {self.uF.item():.4f}")
            logging.info(f"Feature quantizer output_scale updated: {old_scale:.4f} -> {self.output_scale.item():.4f}")

        x = (x - self.lF) / (self.uF - self.lF)
        x = x.clamp(min=0, max=1)

        if not self.baseline:
            x = self.EWGS_discretizer(x, self.num_levels, self.bkwd_scaling_factorF, save_dict, None, None)
        else:
            x = self.STE_discretizer(x, self.num_levels)

        if self.hook_feature_values:
            self.buff_feature = x
            self.buff_feature.retrain_grad()

        x = x * self.output_scale

        return x