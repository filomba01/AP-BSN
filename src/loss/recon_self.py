import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from . import regist_loss


eps = 1e-6

# ============================ #
#  Self-reconstruction loss    #
# ============================ #

@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.l1_loss(output, target_noisy)  

@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output, target_noisy)

# ============================ #
#  Enhanced Detail-Preserving  #
#      Loss Functions          #
# ============================ #

@regist_loss
class self_L1_gradient():
    """L1 loss with gradient penalty to preserve edges and details"""
    def __init__(self, gradient_weight=0.1):
        self.gradient_weight = gradient_weight
    
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        
        # Basic L1 loss
        l1_loss = F.l1_loss(output, target_noisy)
        
        # Gradient loss for edge preservation
        grad_x_out = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])
        grad_y_out = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
        
        grad_x_target = torch.abs(target_noisy[:, :, :, :-1] - target_noisy[:, :, :, 1:])
        grad_y_target = torch.abs(target_noisy[:, :, :-1, :] - target_noisy[:, :, 1:, :])
        
        grad_loss = F.l1_loss(grad_x_out, grad_x_target) + F.l1_loss(grad_y_out, grad_y_target)
        
        return l1_loss + self.gradient_weight * grad_loss

@regist_loss
class self_perceptual():
    """Perceptual loss using VGG features to preserve semantic details"""
    def __init__(self, feature_weight=0.1):
        self.feature_weight = feature_weight
        
        if not TORCHVISION_AVAILABLE:
            print("Warning: torchvision not available, perceptual loss will fallback to L1")
            self.use_perceptual = False
            return
        
        self.use_perceptual = True
        # Load pre-trained VGG19 and freeze parameters
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[:9],   # relu2_2
            vgg[:18],  # relu3_4
            vgg[:27],  # relu4_4
        ])
        
        for layer in self.vgg_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Move to device (will be handled automatically)
        self.device = None
    
    def _get_features(self, x):
        # Normalize for VGG (ImageNet preprocessing)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if self.device is None:
            self.device = x.device
            mean = mean.to(self.device)
            std = std.to(self.device)
            for layer in self.vgg_layers:
                layer = layer.to(self.device)
        
        x_norm = (x - mean) / std
        
        features = []
        for layer in self.vgg_layers:
            x_norm = layer(x_norm)
            features.append(x_norm)
        return features
    
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        
        # Basic L1 loss
        l1_loss = F.l1_loss(output, target_noisy)
        
        # Perceptual loss (if available)
        if self.use_perceptual and output.size(1) == 3:  # RGB images
            output_features = self._get_features(output)
            target_features = self._get_features(target_noisy)
            
            perceptual_loss = 0
            for out_feat, target_feat in zip(output_features, target_features):
                perceptual_loss += F.mse_loss(out_feat, target_feat)
            
            return l1_loss + self.feature_weight * perceptual_loss
        else:
            return l1_loss

@regist_loss
class self_ssim_l1():
    """Combined SSIM and L1 loss for better structural preservation"""
    def __init__(self, ssim_weight=0.5, window_size=11):
        self.ssim_weight = ssim_weight
        self.window_size = window_size
    
    def _ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size//2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size//2)
        
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        
        sigma_x = F.avg_pool2d(x * x, self.window_size, 1, self.window_size//2) - mu_x_sq
        sigma_y = F.avg_pool2d(y * y, self.window_size, 1, self.window_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1, self.window_size//2) - mu_x_mu_y
        
        ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))
        
        return ssim_map.mean()
    
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        
        l1_loss = F.l1_loss(output, target_noisy)
        ssim_loss = 1 - self._ssim(output, target_noisy)
        
        return (1 - self.ssim_weight) * l1_loss + self.ssim_weight * ssim_loss

@regist_loss 
class self_multiscale_l1():
    """Multi-scale L1 loss to capture details at different resolutions"""
    def __init__(self, scales=[1, 0.5, 0.25]):
        self.scales = scales
    
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        
        total_loss = 0
        for scale in self.scales:
            if scale == 1:
                scaled_output = output
                scaled_target = target_noisy
            else:
                h, w = output.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_output = F.interpolate(output, size=(new_h, new_w), mode='bilinear', align_corners=False)
                scaled_target = F.interpolate(target_noisy, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            total_loss += F.l1_loss(scaled_output, scaled_target)
        
        return total_loss / len(self.scales)
