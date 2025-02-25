import torch
import torch.nn as nn
import math
from .custom_modules import STE_discretizer, EWGS_discretizer

class FeatureQuantizer(nn.Module):
    def __init__(self, num_levels, scaling_factor, baseline=False):
        super(FeatureQuantizer, self).__init__()
        self.num_levels = num_levels
        self.scaling_factor = scaling_factor
        self.baseline = baseline

        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
        
        self.uF = nn.Parameter(torch.tensor(1.0))  
        self.lF = nn.Parameter(torch.tensor(0.0))  
        
        self.register_buffer('bkwd_scaling_factorF', torch.tensor(self.scaling_factor).float())
        
        self.hook_feature_values = False
        self.buff_feature = None
        
        self.register_buffer('init', torch.tensor([1])) 
        self.output_scale = nn.Parameter(torch.tensor(1.0, requires_grad=True))  

        
    def forward(self, feature, save_dict=None):
        if self.init == 1:
            self.uF.data.fill_(feature.std() / math.sqrt(1 - 2/math.pi) * 3.0)
            self.lF.data.fill_(feature.min())
            
        feature = (feature - self.lF) / (self.uF - self.lF)
        feature = feature.clamp(min=0, max=1)

        if self.baseline:
            quantized_feature = self.STE_discretizer(feature, self.num_levels)
        else:
            quantized_feature = self.EWGS_discretizer(
                feature, self.num_levels, self.scaling_factor, save_dict, None, None
            )
        
        if self.hook_feature_values:
            self.buff_feature = quantized_feature
            self.buff_feature.retain_grad()

        if self.init == 1:
            self.output_scale.data.fill_(feature.abs().mean() / quantized_feature.abs().mean())
            self.init.data.fill_(0)
        
        quantized_feature *= self.output_scale  
        
        return quantized_feature