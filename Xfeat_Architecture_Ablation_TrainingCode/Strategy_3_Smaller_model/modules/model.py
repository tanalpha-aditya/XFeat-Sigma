"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Modified for Ablation Study (ii): Smaller Model
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)

class XFeatModel(nn.Module):
	"""
	   Implementation of architecture described in
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."

	   *** MODIFIED FOR SMALLER MODEL ABLATION (halved channels in last 3 blocks) ***
	"""

	def __init__(self):
		super().__init__()
		self.norm = nn.InstanceNorm2d(1)


		########### ⬇️ CNN Backbone & Heads ⬇️ ###########

		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

		# Blocks 1 and 2 remain unchanged
		self.block1 = nn.Sequential(
										BasicLayer( 1,  4, stride=1),
										BasicLayer( 4,  8, stride=2),
										BasicLayer( 8,  8, stride=1),
										BasicLayer( 8, 24, stride=2),
									)

		self.block2 = nn.Sequential(
										BasicLayer(24, 24, stride=1),
										BasicLayer(24, 24, stride=1),
									 )

		# --- MODIFICATION START ---
		# Block 3: Output channels halved from 64 to 32
		self.block3 = nn.Sequential(
										BasicLayer(24, 32, stride=2), # Changed 64 -> 32
										BasicLayer(32, 32, stride=1), # Changed 64 -> 32
										BasicLayer(32, 32, 1, padding=0), # Changed 64 -> 32
									 )
		# Block 4: Input and Output channels halved from 64 to 32
		self.block4 = nn.Sequential(
										BasicLayer(32, 32, stride=2), # Changed 64 -> 32 (both in/out)
										BasicLayer(32, 32, stride=1), # Changed 64 -> 32 (both in/out)
										BasicLayer(32, 32, stride=1), # Changed 64 -> 32 (both in/out)
									 )

		# Block 5: Input halved (32), intermediate halved (64), output halved (32)
		self.block5 = nn.Sequential(
										BasicLayer( 32, 64, stride=2), # Input 32 (was 64), Output 64 (was 128)
										BasicLayer( 64, 64, stride=1), # Input 64 (was 128), Output 64 (was 128)
										BasicLayer( 64, 64, stride=1), # Input 64 (was 128), Output 64 (was 128)
										BasicLayer( 64, 32, 1, padding=0), # Input 64 (was 128), Output 32 (was 64)
									 )

		# Block Fusion: Input and Output channels halved from 64 to 32
		self.block_fusion =  nn.Sequential(
										BasicLayer(32, 32, stride=1), # Changed 64 -> 32
										BasicLayer(32, 32, stride=1), # Changed 64 -> 32
										nn.Conv2d (32, 32, 1, padding=0) # Changed 64 -> 32
									 )

		# Heatmap Head: Input channels halved from 64 to 32
		self.heatmap_head = nn.Sequential(
										BasicLayer(32, 32, 1, padding=0), # Changed 64 -> 32
										BasicLayer(32, 32, 1, padding=0), # Changed 64 -> 32
										nn.Conv2d (32, 1, 1), # Input Changed 64 -> 32
										nn.Sigmoid()
									)

		# Keypoint Head: Operates on unfolded input, kept unchanged as per interpretation
		self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)


  		########### ⬇️ Fine Matcher MLP ⬇️ ###########
        # Input halved (concat(32, 32) = 64), intermediate layers halved (256), output remains 64 (for 8x8 grid)
		self.fine_matcher =  nn.Sequential(
											nn.Linear(64, 256), # Input changed 128->64, Output changed 512->256
											nn.BatchNorm1d(256, affine=False), # Changed 512->256
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 256), # Changed 512->256
											nn.BatchNorm1d(256, affine=False), # Changed 512->256
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 256), # Changed 512->256
											nn.BatchNorm1d(256, affine=False), # Changed 512->256
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 256), # Changed 512->256
											nn.BatchNorm1d(256, affine=False), # Changed 512->256
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 64), # Output size MUST remain 64 for offset logits
										)
        # --- MODIFICATION END ---

	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 32, H/8, W/8) dense local features (MODIFIED: 64->32 channels)
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map (Unchanged)
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map (Unchanged shape, but derived from smaller features)

		"""
		#dont backprop through normalization
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		#main backbone
		x1 = self.block1(x)
		x2 = self.block2(x1 + self.skip1(x))
		x3 = self.block3(x2) # Now outputs 32 channels
		x4 = self.block4(x3) # Now outputs 32 channels
		x5 = self.block5(x4) # Now outputs 32 channels

		#pyramid fusion
		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear') # Interpolates 32 channels
		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear') # Interpolates 32 channels
		# Input to fusion is sum of 32-channel tensors
		feats = self.block_fusion( x3 + x4 + x5 ) # Now outputs 32 channels

		#heads
		heatmap = self.heatmap_head(feats) # Takes 32 channels input
		keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) # Unchanged head path

		return feats, keypoints, heatmap # Note: feats is now 32-dimensional
