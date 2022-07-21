import torch.nn as nn
  
class RmlModel(nn.Module):
    def __init__(self):
        super(RmlModel, self).__init__()

    def is_validation_step_available(self):
        return hasattr(self, 'validation_step')
        
    def is_before_evaluation_available(self):
        return hasattr(self, 'before_evaluation')
        
    def is_after_evaluation_available(self):
        return hasattr(self, 'after_evaluation')
