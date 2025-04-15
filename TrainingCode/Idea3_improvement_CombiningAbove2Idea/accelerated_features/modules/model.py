import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLayer(nn.Module):
    """
      Basic Convolutional Layer: Conv2d -> Normalization -> ReLU -> Optional Spatial Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 use_dropout=False, dropout_prob=0.3, norm_type='batch'):
        super().__init__()
        # Choose normalization type
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(out_channels)
        elif norm_type == 'group':
            # Using 4 groups here as an example; adjust based on channel count
            norm_layer = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        else:
            raise ValueError("Unsupported norm_type. Use 'batch' or 'group'.")

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            norm_layer,
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            # Using spatial dropout for convolutional features
            layers.append(nn.Dropout2d(p=dropout_prob))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
        
        
# CBAM: Convolutional Block Attention Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention components
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x


class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Replace InstanceNorm2d with GroupNorm for improved stability on small batches.
        # Since input is single channel, we can use a GroupNorm with one channel per group,
        # but here we'll normalize after converting to a feature map.
        # For the input normalization, we'll keep it simple and use BatchNorm for demonstration.
        self.norm = nn.BatchNorm2d(1)

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )

        # In the following blocks, we add optional dropout and use our BasicLayer with GroupNorm.
        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(4, 8, stride=2, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(8, 8, stride=1, norm_type='group'),
            BasicLayer(8, 24, stride=2, norm_type='group')
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(24, 24, stride=1, norm_type='group')
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(64, 64, stride=1, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(64, 64, kernel_size=1, padding=0, norm_type='group')
        )

        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(64, 64, stride=1, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(64, 64, stride=1, norm_type='group')
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(128, 128, stride=1, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(128, 128, stride=1, norm_type='group'),
            BasicLayer(128, 64, kernel_size=1, padding=0, norm_type='group')
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1, norm_type='group'),
            BasicLayer(64, 64, stride=1, norm_type='group'),
            nn.Conv2d(64, 64, 1, padding=0)
        )

        # Add CBAM after feature fusion
        self.cbam = CBAM(channels=64)
        
        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, kernel_size=1, padding=0, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(64, 64, kernel_size=1, padding=0, norm_type='group'),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, kernel_size=1, padding=0, use_dropout=True, dropout_prob=0.3, norm_type='group'),
            BasicLayer(64, 64, kernel_size=1, padding=0, norm_type='group'),
            BasicLayer(64, 64, kernel_size=1, padding=0, norm_type='group'),
            nn.Conv2d(64, 65, 1)
        )

        ########### ⬇️ Fine Matcher MLP ⬇️ ###########
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64)
        )

    def _unfold2d(self, x, ws=2):
        """
            Unfolds tensor in 2D with desired window size (ws) and concatenates channels.
        """
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws) \
            .reshape(B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        """
            x: torch.Tensor(B, C, H, W) - grayscale or RGB images.
            Returns:
              feats     -> torch.Tensor(B, 64, H/8, W/8) dense local features.
              keypoints -> torch.Tensor(B, 65, H/8, W/8) keypoint logit map.
              heatmap   -> torch.Tensor(B, 1, H/8, W/8) reliability map.
        """
        # Average over channels and apply normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # Pyramid fusion with interpolation to match dimensions
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)
        
        # Apply attention mechanism on fused features
        feats = self.cbam(feats)

        heatmap = self.heatmap_head(feats)  # Reliability map head
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))  # Keypoint map logits

        return feats, keypoints, heatmap

