# Copyright 2024 Shenyang Institute of Automation, Chinese Academy of Sciences
# Licensed under the MIT License. See the LICENSE file for details.

# The following code is licensed under the Apache License 2.0:
#
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note: Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

from .cudalight._torch_impl import quat_to_rotmat  # Function to convert from quaternion to rotation matrix
from .rasterizerlight.project_gaussians import project_gaussians  # Projection Gaussian function
from .rasterizerlight.rasterize import rasterize_gaussians
from .utils.sh import num_sh_bases, spherical_harmonics  # Spherical harmonics related functions

from pytorch_msssim import SSIM
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import SHEncoding
import torch.nn.functional as F
import kornia

from .losses import DarkChannelPriorLossV3, EdgeSimilarityLoss, GrayWorldPriorLoss, exposure_loss



def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@dataclass
class LowlightUnderwaterModelConfig(ModelConfig):

    """lowlight underwater Model Config"""


    _target: Type = field(default_factory=lambda: LowlightUnderwaterModel)
    num_steps: int = 15000
    """Number of steps to train the model"""
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.5 # Impact distribution, choose a lower value for defogging, too high will make everything disappear, and you can use a high value of 0.5 for underwater.
    """opacity threshold for culling Gaussians. Can be set to lower value (e.g. 0.005) to get higher quality"""
    cull_alpha_thresh_post: float = 0.05 # 正常光水下用高的可以0.1，
    """opacity threshold of Gaussian after culling Threshold of late stage"""
    reset_alpha_thresh: float = 0.5
    """reset opacity threshold"""
    cull_scale_thresh: float = 10.0 # 10.0 would be better？
    """scale threshold to remove huge Gaussians"""
    continue_cull_post_densification: bool = True
    """if True, continue to clip Gaussian points after thinning"""
    zero_medium: bool = False
    """if True, close medium network"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """if True, use absolute gradients for densification"""
    densify_grad_thresh: float = 0.0008 #影响分布
    """position gradient norm threshold for dense Gaussian distribution （0.0004， 0.0008）"""
    use_absgrad: bool = False # False？
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    clip_thresh: float = 0.01
    """minimum depth threshold"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 10000
    """stop splitting at this step"""
    sh_degree: int = 1
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False# False better? Reduce Gaussian splitting of large volumes
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic rendering mode uses EWA convolution volume splatting with a screen-space blur kernel of [0.3, 0.3].
    However, this approach is not suitable for rendering very small gaussians at high or low resolutions, causing artifacts similar to the "aliasing" effect.
    Antialiased mode overcomes this limitation by calculating a compensation factor and applying it to the transparency of the gaussian, maintaining the integral of the total splatting density.
    However, PLY files exported using the antialiased splatting mode are not compatible with Classic mode. As a result, many web viewers implemented for Classic mode cannot correctly render antialiased PLY files without modification.
    """
    num_layers_medium: int = 2
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium MLP."""
    medium_density_bias: float = 0.0
    num_layers_color: int = 2
    """Number of hidden layers for color MLP."""
    hidden_dim_color: int = 128
    """Dimension of hidden layers for color MLP."""
    color_density_bias: float = 0.0
    """Bias for color density (sigma_bs and sigma_attn)."""
    mlp_type: Literal["tcnn", "torch"] = "tcnn"
    mlp_type2: Literal["tcnn", "torch"] = "tcnn"
    mlp_type3: Literal["tcnn", "torch"] = "tcnn"
    """Type of MLP to use for medium MLP."""
    gamma_0: float = 2.2
    color_mlp_out_dim: int = 1
    """Number of illumination channels: 1 shares illumination across RGB; 3 uses independent channels."""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SE3"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """if True, apply color correction to the rendered images before computing the metrics."""
    one_color: bool = False
    """if True, close enhancement network"""
    enhance_enable: bool = False
    """Enable Enhancement, dimming must be True"""
    alpha_per_channel: bool = False
    """Use three enhancement scale channels instead of one shared scale."""
    white_balance_loss: bool = False
    """Enable white balance loss"""
    bs_loss_enable: bool = False
    """Enable bs loss to control the scattered component (medium)"""
    gray_loss_enable: bool = False
    """Master switch for the LLNeRF prior bundle (gray world + exposure +
    edge). When True, all three are activated after step 3000. The
    sub-flags below let you disable individual components — useful for
    underwater scenes where the gray-world assumption (channels balanced
    after correction) conflicts with the physics (water selectively
    attenuates R/G/B)."""
    gray_world_loss_enable: bool = True
    """Enable the gray-world component when gray_loss_enable is active."""
    edge_prior_loss_enable: bool = True
    """Enable the edge-alignment component when gray_loss_enable is active."""
    exposure_prior_loss_enable: bool = True
    """Enable the exposure component when gray_loss_enable is active."""
    water_splatting_loss: bool = False
    """Use the underwater reconstruction loss."""
    wb_clamp: float = 7
    wb_anchor_loss: bool = False
    """Anchor the enhanced image to a detached white-balanced target after warm-up."""
    wb_anchor_weight: float = 0.5
    """Weight for wb_anchor_loss (absolute L1 scale)."""
    wb_anchor_warmup: int = 3000
    """Anchor only kicks in after this step. Same warm-up as gray_loss_enable."""
    wb_anchor_mask_thresh: float = 0.5
    """Minimum visibility for white-balance anchoring; set to zero to disable masking."""
    clear_sparsity_loss: bool = False
    """Penalize the clear rendering in regions with low visibility."""
    clear_sparsity_weight: float = 0.01
    """Weight for clear_sparsity_loss."""
    opacity_l1_loss: bool = False
    """Apply an L1 penalty to Gaussian opacities."""
    opacity_l1_weight: float = 1.0e-4
    """Weight for opacity_l1_loss."""
    alpha_decay_loss: bool = False
    """Penalize accumulated opacity in regions with low visibility."""
    alpha_decay_weight: float = 0.01
    """Weight for alpha_decay_loss."""
    sigma_order_loss: bool = False
    """Penalize attenuation coefficients that violate the ordering R >= G >= B."""
    sigma_order_weight: float = 0.01
    """Weight for sigma_order_loss."""
    encode_recon_loss: bool = False
    """Apply the configured gamma encoding before computing the reconstruction loss."""
    encode_gamma: float = 2.2
    """Gamma value for the encoding (only used when encode_recon_loss is True).
    The encode is x ** (1 / encode_gamma)."""
    scb_gray_world_loss: bool = False
    """Penalize channel means relative to a detached affine target."""
    scb_gray_world_weight: float = 0.1
    """Weight for scb_gray_world_loss."""
    scb_gray_world_target_base: float = 0.35
    """Base in target_c = base + lambda * mean_c.detach()."""
    scb_gray_world_lambda: float = 0.14
    """Mixing coefficient in the detached per-channel target."""
    chart_supervised_loss: bool = False
    """Enable optional RGB supervision at annotated color-chart patch centers."""
    chart_supervised_weight: float = 0.5
    """Weight of the color-chart RGB L1 loss."""
    chart_supervised_warmup: int = 3000
    """Training step after which color-chart supervision starts."""
    chart_supervised_patch_radius: int = 3
    """Half-window for the patch sample on the rendered image (downscaled
    coords). 3 → 7×7 window, robust to sub-pixel mis-registration."""
    chart_supervision_path: str = ""
    """Optional path to color-chart annotations. An empty path disables chart supervision."""
    chart_supervision_target: Literal["clear_enhanced", "enhanced"] = "clear_enhanced"
    """Render path for chart supervision: clear_enhanced or enhanced."""


class LowlightUnderwaterModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: LowlightUnderwaterModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.xys_grad_norm = None
        self.sigma_activation = None
        self.colour_activation = None
        self.medium_density_bias = None
        self.color_density_bias = None
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        # initialize the medium MLP
        self.direction_encoding = SHEncoding(levels=4, implementation="tcnn")
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus() # relu
        # medium MLP hyper-params
        num_layers_medium = self.config.num_layers_medium
        hidden_dim_medium = self.config.hidden_dim_medium
        self.medium_density_bias = self.config.medium_density_bias
        # color / enhance MLP hyper-params
        num_layers_color = self.config.num_layers_color
        hidden_dim_color = self.config.hidden_dim_color
        self.color_density_bias = self.config.color_density_bias

        # ------------------------medium network------------------------
        if num_layers_medium > 1:
            self.medium_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim(),
                num_layers=num_layers_medium,
                layer_width=hidden_dim_medium,
                out_dim=9,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=self.config.mlp_type,
            )
        else:
            self.medium_mlp = nn.Linear(self.direction_encoding.get_out_dim(), 9)
            self.config.mlp_type = "torch"
        # ------------------------color network (per-pixel illumination L)------------------------
        # LLNeRF: L is single-channel; R lives in the gaussian SH colors (3-ch).
        color_mlp_out_dim = self.config.color_mlp_out_dim
        if num_layers_color > 1:
            self.color_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim(),
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=color_mlp_out_dim,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=self.config.mlp_type2,
            )
        else:
            self.color_mlp = nn.Linear(self.direction_encoding.get_out_dim(), color_mlp_out_dim)
            self.config.mlp_type2 = "torch"

        # ------------------------enhance MLP (LLNeRF style: gamma 3-ch + alpha 1 or 3-ch)------------------------
        # output 4 (default) or 6 (alpha_per_channel) channels:
        #   [..., :3] -> gamma_coeff (always per-channel),
        #   [..., 3:4] or [..., 3:6] -> alpha_coeff (1 or 3 ch).
        enhance_in_dim = self.direction_encoding.get_out_dim() + color_mlp_out_dim
        enhance_out_dim = 6 if self.config.alpha_per_channel else 4
        if num_layers_color > 1:
            self.enhance_mlp = MLP(
                in_dim=enhance_in_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=enhance_out_dim,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=self.config.mlp_type3,
            )
        else:
            self.enhance_mlp = nn.Linear(enhance_in_dim, enhance_out_dim)
            self.config.mlp_type3 = "torch"


        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        # calculate the nearest neighbor distance
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale

        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points)) # random rotation
        dim_sh = num_sh_bases(self.config.sh_degree)


        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            # NOTE: must use .cuda() rather than .to(self.device) here. populate_modules()
            # runs before nn.Module registers device_indicator_param, so self.device would
            # raise AttributeError. Splatfacto's upstream uses .cuda() for the same reason.
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255) # Convert color to SH
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :]) # direct color characteristics of the low-frequency part of the sneakers, the 0th-order component of the spherical harmonics
            features_rest = torch.nn.Parameter(shs[:, 1:, :]) # higher-order components (from 1st to higher)
        else:

            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))


        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup( #closed optimizer?
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        # loss criteria (created once, moved to device in populate_modules)
        if self.config.bs_loss_enable:
            self.dcp_criterion = DarkChannelPriorLossV3()
        if self.config.gray_loss_enable:
            self.edge_criterion = EdgeSimilarityLoss()
            self.gray_criterion = GrayWorldPriorLoss()

        # Load and sample optional per-frame color-chart supervision in native image coordinates.
        self._chart_supervision: Optional[Dict] = None
        self._train_image_stems: Optional[List[str]] = None
        if self.config.chart_supervised_loss and self.config.chart_supervision_path:
            import json as _json
            from pathlib import Path as _Path
            cs_path = _Path(self.config.chart_supervision_path)
            if cs_path.is_file():
                cs = _json.loads(cs_path.read_text())
                # Pre-build a [18,3] sRGB-in-[0,1] reference tensor.
                ref_255 = torch.tensor(
                    cs["_meta"]["ref_rgb_srgb_255"], dtype=torch.float32
                )
                self._chart_ref_rgb = (ref_255 / 255.0).cuda()  # [18, 3]
                self._chart_supervision = cs
                CONSOLE.log(
                    f"[chart_supervised_loss] loaded {len(cs['frames'])} frames "
                    f"from {cs_path}"
                )
            else:
                CONSOLE.log(
                    f"[chart_supervised_loss] WARN: path not found, loss disabled: {cs_path}"
                )


        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":

            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:

            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]
    
    # medium_mlp, color_mlp, enhance_mlp, direction_encoding are set as
    # direct instance attributes in populate_modules(), not in gauss_params.

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.num_steps
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.

        Args:
            x (torch.Tensor): The data tensor of shape [num_samples, num_features]
            k (int): The number of neighbors to retrieve

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of the k-nearest neighbors
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Determine the number of samples
        n_samples = x_np.shape[0]

        # Calculate the number of neighbors, ensuring it does not exceed n_samples
        n_neighbors = min(k + 1, n_samples)

        if n_neighbors < k + 1:
            print(f"Warning: Adjusting n_neighbors from {k + 1} to {n_neighbors} because n_samples={n_samples}")

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        if n_neighbors > 1:
            # Exclude the point itself from the result and return
            return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
        else:
            # Only one sample, no neighbors
            return distances[:, 0:0].astype(np.float32), indices[:, 0:0].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                grads = self.xys_grad_abs.detach().norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all opacity reset logic by refine_every
            # This many steps so we don't save checkpoints on opacity resets (every 2000 steps)
            # Then cull. Split/cull only if we've seen every image since the last opacity reset.
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and (self.step % reset_interval > self.num_train_data + self.config.refine_every)
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])

                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # if self.step < self.config.stop_screen_size_at:
                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )                
                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None
    
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)
                # NB: do NOT reset Adam exp_avg/exp_avg_sq for the auxiliary
                # MLPs (medium / color / enhance / direction_encoding). Their
                # tensor shapes never change; only the gaussian param shapes
                # change after split/dup/cull. Resetting MLP momentum every
                # 100 steps was effectively turning Adam into SGD and starved
                # enhance_mlp of progress (A1 showed it stuck near init).

            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:                
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            
            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        # 裁剪不透明度低于阈值的高斯点
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []#maybe this have some probrom
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],self.step_cb, args=[training_callback_attributes.optimizers],))
        # cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # after each training iteration, execute after_train
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
    # def step_cb(self, step):
    #     self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
            "gaussian": [...],
            "medium_mlp": [...],
            "color_mlp": [...],
            "enhance_mlp": [...],
            "direction_encoding": [...]
        """
        gps = self.get_gaussian_param_groups()
        gps["medium_mlp"] = list(self.medium_mlp.parameters())
        # print(gps["medium_mlp"])
        gps["color_mlp"] = list(self.color_mlp.parameters())
        gps["enhance_mlp"] = list(self.enhance_mlp.parameters())
        gps["direction_encoding"] = list(self.direction_encoding.parameters())
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    def get_outputs(self, camera: Cameras,obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.
            obb_box (Optional[OrientedBox], optional):

        Returns:
            Dict[str, Union[torch.Tensor, List]]:
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        camera_downscale = self._get_downscale_factor()# downsample 4 times
        camera.rescale_output_resolution(1 / camera_downscale)

        R = camera.camera_to_worlds[0, :3, :3]  # 3x3 rotation matrix
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3x1 translation vector

        # flip the z and y axes to align with gsplat's conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit  # apply flip

        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the camera's field of view
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        self.last_fx = camera.fx.item()
        self.last_fy = camera.fy.item()

        # media section
        # pixel encoding
        y = torch.linspace(0., H, H, device=self.device)
        x = torch.linspace(0., W, W, device=self.device)
        yy, xx = torch.meshgrid(y, x)
        yy = (yy - cy) / camera.fy.item()
        xx = (xx - cx) / camera.fx.item()
        directions = torch.stack([yy, xx, -1 * torch.ones_like(xx)], dim=-1)
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        directions = directions / norms
        directions = directions @ R

        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat) # Spherical harmonic encoding, after encoding is H*W,16
        outputs_shape = directions.shape[:-1]


        if self.config.mlp_type == "tcnn":
            medium_base_out = self.medium_mlp(directions_encoded)
        else:
            medium_base_out = self.medium_mlp(directions_encoded.float())


        medium_rgb = (#[H,W,3] CLAMP(0-1)
            self.colour_activation(medium_base_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )# apply Sigmoid activation to the color part
        medium_bs = (#[H,W,3] CLAMP(>0)
            self.sigma_activation(medium_base_out[..., 3:6] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )# sigma_bs part applies Sigmoid activation
        medium_attn = (#[H,W,3] CLAMP(>0)
            self.sigma_activation(medium_base_out[..., 6:] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )# sigma_attn part applies Sigmoid activation

        if self.config.zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)


        if self.config.mlp_type2 == "tcnn":
            color_base_out = self.color_mlp(directions_encoded)
        else:
            color_base_out = self.color_mlp(directions_encoded.float())

        # Predict illumination with the configured number of channels.
        color_mlp_out_dim = self.config.color_mlp_out_dim
        color_rgb = (
            self.colour_activation(color_base_out[..., :color_mlp_out_dim])
            .view(*outputs_shape, -1)
            .to(directions)
        )
        color_rgb = torch.clamp(color_rgb, min=1e-4, max=1.0)

        # LLNeRF-style enhancement: alpha is a single illumination scaler,
        # gamma is per-channel. Both are computed from a stop-gradient view
        # of color_rgb so the reconstruction loss does not contaminate them.
        if self.config.enhance_enable and not self.config.one_color:
            color_rgb_sg = color_rgb.detach().view(-1, color_rgb.shape[-1])
            if self.config.mlp_type3 == "tcnn":
                enhance_base_out = self.enhance_mlp(
                    torch.cat([directions_encoded, color_rgb_sg], dim=-1)
                )
            else:
                enhance_base_out = self.enhance_mlp(
                    torch.cat([directions_encoded, color_rgb_sg], dim=-1).float()
                )

            gamma_coef = (  # [H,W,3] in (0,1) — per-channel gamma_coeff
                self.colour_activation(enhance_base_out[..., :3])
                .view(*outputs_shape, -1)
                .to(directions)
            )
            alpha_end = 6 if self.config.alpha_per_channel else 4
            alpha_coef = (  # [H,W,1] or [H,W,3] in (0,1) — illumination scaler
                self.colour_activation(enhance_base_out[..., 3:alpha_end])
                .view(*outputs_shape, -1)
                .to(directions)
            )
            gamma_coef = torch.clamp(gamma_coef, min=1e-4, max=1.0)
            alpha_coef = torch.clamp(alpha_coef, min=1e-4)
            final_gamma = torch.clamp(1 / (gamma_coef + self.config.gamma_0), min=0.1, max=5)

            # stop-gradient on color_rgb: enhancement coefficients are driven
            # only by the prior losses, never by reconstruction.
            enhanced_color_rgb = (color_rgb.detach() / (alpha_coef + 1e-4)) ** final_gamma
            # Use a shared illumination scale when per-channel scaling is disabled.
            enhanced_color_rgb = torch.clamp(enhanced_color_rgb, min=0.0, max=3.0)
        else:
            # enhancement disabled: bypass enhance_mlp, use unit modulation.
            gamma_coef = torch.zeros((*outputs_shape, 3), device=directions.device, dtype=directions.dtype)
            alpha_coef = torch.ones((*outputs_shape, 1), device=directions.device, dtype=directions.dtype)
            enhanced_color_rgb = torch.ones((*outputs_shape, 3), device=directions.device, dtype=directions.dtype)

        if self.config.one_color:
            color_rgb = torch.ones_like(color_rgb)

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = medium_rgb
                depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                zero_obj = torch.zeros_like(rgb)
                return {
                    "rgb": rgb, "rgb_lowlight": rgb, "rgb_enhanced": rgb,
                    "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                    "rgb_object": zero_obj, "rgb_object_enhanced": zero_obj,
                    "rgb_clear": zero_obj, "rgb_clear_enhanced": zero_obj,
                    "rgb_medium": medium_rgb, "pred_image": rgb,
                    "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn,
                    "color_rgb": color_rgb, "enhanced_color_rgb": enhanced_color_rgb,
                    "alpha_coef": alpha_coef, "gamma_coef": gamma_coef,
                }
        else:
            crop_ids = None

        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)# Combination shoes need to be always changed to 0 stage components and compensated with the results of mlp
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            clip_thresh=self.config.clip_thresh,
        )  # type: ignore

        self.depths = depths.detach()

        # Rescale the camera back to its original size before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            zero_obj = torch.zeros_like(rgb)
            return {
                "rgb": rgb, "rgb_lowlight": rgb, "rgb_enhanced": rgb,
                "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                "rgb_object": zero_obj, "rgb_object_enhanced": zero_obj,
                "rgb_clear": zero_obj, "rgb_clear_enhanced": zero_obj,
                "rgb_medium": medium_rgb, "pred_image": rgb,
                "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn,
                "color_rgb": color_rgb, "enhanced_color_rgb": enhanced_color_rgb,
                "alpha_coef": alpha_coef, "gamma_coef": gamma_coef,
            }

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        self.xys_grad_abs = torch.zeros_like(self.xys)

        # Render with unit color modulation. The CUDA path therefore returns
        #   rgb_object_low = Σ vis * c_gauss * exp(-σ_attn * d)         (D, low-light)
        #   rgb_clear_low  = Σ vis * c_gauss                            (J, low-light)
        # Per-pixel enhancement (and any per-pixel color modulation) is
        # multiplied in Python so gradients w.r.t. enhancement coefficients
        # flow only through the prior losses, never through the CUDA backward.
        unit_modulation = torch.ones_like(medium_rgb)
        rgb_object_low, rgb_clear_low, rgb_medium, depth_im, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            self.xys_grad_abs,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            medium_rgb,
            medium_bs,
            medium_attn,
            unit_modulation,
            H,
            W,
            BLOCK_WIDTH,
            background=medium_rgb,
            return_alpha=True,
            step=self.step,
        )  # type: ignore
        rgb_object_low = torch.clamp(rgb_object_low, 0, 1)
        rgb_clear_low = torch.clamp(rgb_clear_low, 0., 1.)

        # ----- LLNeRF L-R style decomposition -----
        # gaussian SH color = R (per-3D-point reflectance)
        # color_rgb         = L (per-pixel illumination, view-dependent)
        # enhanced_color_rgb = L_e (illumination after (sg(L)/alpha)^gamma)
        if self.config.enhance_enable and not self.config.one_color:
            # Low-light path: L * R (with attenuation), matched to captured GT.
            rgb_clear_LR = rgb_clear_low * color_rgb
            rgb_object_LR = rgb_object_low * color_rgb
            rgb_lowlight = torch.clamp(rgb_object_LR + rgb_medium, 0, 1)
            # Enhanced path: L_e * R, matched only by priors.
            rgb_clear_enhanced = rgb_clear_low * enhanced_color_rgb
            rgb_object_enhanced = rgb_object_low * enhanced_color_rgb
            rgb_enhanced = rgb_object_enhanced + rgb_medium
            # Also expose the un-enhanced J = L * R for DCP / metrics.
            rgb_clear_for_dcp = rgb_clear_LR
        else:
            # Enhancement disabled: keep behavior of legacy yamls — single path,
            # gaussian SH color is the only reflectance, no per-pixel modulation.
            rgb_lowlight = torch.clamp(rgb_object_low + rgb_medium, 0, 1)
            rgb_clear_enhanced = rgb_clear_low
            rgb_object_enhanced = rgb_object_low
            rgb_enhanced = rgb_lowlight
            rgb_clear_for_dcp = rgb_clear_low

        depth_im = depth_im[..., None]
        alpha = alpha[..., None]
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        return {
            "rgb": rgb_lowlight,                          # main reconstruction output (matches GT)
            "rgb_lowlight": rgb_lowlight,
            "rgb_enhanced": rgb_enhanced,                 # enhanced render (for viewer / metrics)
            "depth": depth_im,
            "accumulation": alpha,
            "background": rgb_medium,
            "rgb_object": rgb_object_low,                 # raw object render before L modulation
            "rgb_object_enhanced": rgb_object_enhanced,
            "rgb_clear": rgb_clear_for_dcp,               # J = L * R for DCP / metrics
            "rgb_clear_enhanced": rgb_clear_enhanced,     # J after L_e/L scaling (used by gray/exposure/edge)
            "rgb_medium": rgb_medium,
            "pred_image": rgb_lowlight,
            "medium_rgb": medium_rgb,
            "medium_bs": medium_bs,
            "medium_attn": medium_attn,
            "color_rgb": color_rgb,
            "enhanced_color_rgb": enhanced_color_rgb,
            "alpha_coef": alpha_coef,                     # [H,W,1] illumination scaler
            "gamma_coef": gamma_coef,                     # [H,W,3] per-channel gamma coeff
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose.

        LLNeRF-style training compares the low-light render against the *raw*
        captured image. The previous shades-of-grey white-balance hack on the
        GT has been removed so the enhancement coefficients (alpha/gamma) are
        driven purely by the prior losses on the enhanced render.
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            # alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            # return alpha * image[..., :3] + (1 - alpha) * background
            return image[..., :3]# important the background blending method of the original Gaussian has been cancelled
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:  # Color correction indicator is true
            cc_rgb = color_correct(predicted_rgb, gt_rgb) # Make the predicted image close to the color of the gt image, and then detect the psnr indicator
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb) #

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
            metrics_dict[f"enhanced_color_rgb{i}"] = outputs["enhanced_color_rgb"][:, :, i].mean()
            metrics_dict[f"gamma_coef_{i}"] = outputs["gamma_coef"][:, :, i].mean()
        # color_rgb (L) may be 1ch or 3ch depending on config.color_mlp_out_dim.
        c_rgb = outputs["color_rgb"]
        for i in range(c_rgb.shape[-1]):
            metrics_dict[f"color_rgb_{i}"] = c_rgb[:, :, i].mean()
        # alpha is 1 or 3 channel illumination scaler (config.alpha_per_channel).
        a_coef = outputs["alpha_coef"]
        metrics_dict["alpha_coef"] = a_coef.mean()
        if a_coef.shape[-1] == 3:
            for i in range(3):
                metrics_dict[f"alpha_coef_{i}"] = a_coef[:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        depth_img = outputs["depth"]
        # Reconstruction is supervised by the LOW-LIGHT path (matches captured GT).
        pred_img = outputs["rgb_lowlight"]
        # Enhancement priors operate on the L_e * R path (no medium, no attenuation).
        rgb_clear_enhanced = outputs["rgb_clear_enhanced"]
        # DCP/backscatter prior still uses the un-enhanced J (low-light reflectance).
        rgb_clear_low = outputs["rgb_clear"]
        gt_img_detach = gt_img.detach()
        rgb_clear_enhanced_detach = rgb_clear_enhanced.detach()
        pred_img_detach = pred_img.detach()
        depth_img_detach = depth_img.detach()

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
        # loss used to reduce Gaussian splitting of large volumes
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        if self.config.white_balance_loss:
            white_balance_image = self.simple_color_balance_tensor(rgb_clear_enhanced_detach).detach()
            l1_white = torch.abs((white_balance_image - rgb_clear_enhanced) / (rgb_clear_enhanced_detach + 1e-3)).mean()
            white_balance_loss = l1_white * 0.1
        else:
            white_balance_loss = torch.tensor(0.0).to(self.device)
        # DCP backscatter prior on low-light J (un-enhanced): structurally constrains
        # the underlying reflectance independent of the illumination factor.
        if self.config.bs_loss_enable:
            bsdcp_loss = self.dcp_criterion(rgb_clear_low) * 0.08
            bs_loss = bsdcp_loss
        else:
            bs_loss = torch.tensor(0.0).to(self.device)
        # Gray-world / exposure / edge priors drive the enhancement (alpha, gamma)
        # ONLY through the enhanced J path. Reconstruction loss does not see them.
        if self.config.gray_loss_enable and self.step > 3000:
            zero = torch.tensor(0.0, device=self.device)
            edge_loss = (
                self.edge_criterion(rgb_clear_enhanced, gt_img_detach) * 0.1
                if self.config.edge_prior_loss_enable else zero
            )
            gray_loss = (
                self.gray_criterion(rgb_clear_enhanced) * 0.1
                if self.config.gray_world_loss_enable else zero
            )
            image_enhance_loss = (
                exposure_loss(rgb_clear_enhanced) * 0.1
                if self.config.exposure_prior_loss_enable else zero
            )
            color_loss = image_enhance_loss + gray_loss + edge_loss
        else:
            color_loss = torch.tensor(0.0).to(self.device)

        # Visibility combines accumulated opacity with medium attenuation.
        with torch.no_grad():
            alpha_accum_detach = outputs["accumulation"].detach()       # [H, W, 1]
            depth_detach = outputs["depth"].detach()                    # [H, W, 1]
            mean_attn = outputs["medium_attn"].detach().mean(           # [H, W, 1]
                dim=-1, keepdim=True
            )
            transmittance = torch.exp(-mean_attn * depth_detach)        # [H, W, 1]
            visibility = (alpha_accum_detach * transmittance).clamp(0., 1.)

        # Form a detached white-balance target from visible pixels.
        if self.config.wb_anchor_loss and self.step > self.config.wb_anchor_warmup:
            with torch.no_grad():
                if self.config.wb_anchor_mask_thresh > 0:
                    mask = (visibility > self.config.wb_anchor_mask_thresh).float()  # [H,W,1]
                else:
                    mask = torch.ones_like(visibility)
                masked_sum = (rgb_clear_enhanced_detach * mask).sum(dim=(0, 1))     # [3]
                count = mask.sum() + 1e-6
                mean_ch = masked_sum / count                                         # [3]
                gray = mean_ch.mean()
                scale = (gray / (mean_ch + 1e-6)).clamp(0.3, 3.0)                    # [3]
                gt_wb = (rgb_clear_enhanced_detach * scale.view(1, 1, 3)).clamp(0., 1.)
            anchor_diff = torch.abs(gt_wb - rgb_clear_enhanced)
            if self.config.wb_anchor_mask_thresh > 0:
                wb_anchor_loss = (mask * anchor_diff).sum() / (mask.sum() * 3 + 1e-6)
            else:
                wb_anchor_loss = anchor_diff.mean()
            wb_anchor_loss = wb_anchor_loss * self.config.wb_anchor_weight
        else:
            wb_anchor_loss = torch.tensor(0.0).to(self.device)

        # J-sparsity: pull rgb_clear toward 0 weighted by (1 - visibility),
        # so floaters whose signal can't reach the camera get their SH
        # darkened to black.
        if self.config.clear_sparsity_loss:
            clear_sparsity_loss = (
                (1.0 - visibility) * rgb_clear_low.abs()
            ).mean() * self.config.clear_sparsity_weight
        else:
            clear_sparsity_loss = torch.tensor(0.0).to(self.device)

        # Global opacity L1: floaters lose to it because nothing else pushes
        # their opacity up; real surface gauss keep their opacity via
        # reconstruction grad. Decayed gauss eventually cross
        # cull_alpha_thresh and get pruned, letting medium MLP take over
        # the regions floaters used to fake.
        if self.config.opacity_l1_loss:
            opacity_l1_loss = (
                torch.sigmoid(self.opacities).mean() * self.config.opacity_l1_weight
            )
        else:
            opacity_l1_loss = torch.tensor(0.0).to(self.device)

        # Visibility-targeted alpha decay: pull alpha_accum toward 0 at
        # low-vis pixels. Backward chain pushes contributing gauss opacity
        # down. alpha_accum is grad-tracking (rasterizer is differentiable),
        # visibility is detached and only used as a per-pixel weight.
        if self.config.alpha_decay_loss:
            alpha_grad = outputs["accumulation"]
            alpha_decay_loss = (
                ((1.0 - visibility) * alpha_grad).mean() * self.config.alpha_decay_weight
            )
        else:
            alpha_decay_loss = torch.tensor(0.0).to(self.device)

        # σ_attn monotonic ordering prior: σ_R >= σ_G >= σ_B (water physics).
        # Selects the physically-correct branch of the (σ, c_gauss, medium)
        # ambiguity in the SeaThru reconstruction equation, so c_gauss is
        # freed from having to absorb missing R-attenuation as colour cast.
        if self.config.sigma_order_loss:
            sigma_attn = outputs["medium_attn"]   # [H, W, 3]
            sigma_R = sigma_attn[..., 0]
            sigma_G = sigma_attn[..., 1]
            sigma_B = sigma_attn[..., 2]
            sigma_order_loss = (
                F.relu(sigma_G - sigma_R) + F.relu(sigma_B - sigma_G)
            ).mean() * self.config.sigma_order_weight
        else:
            sigma_order_loss = torch.tensor(0.0).to(self.device)

        # Use the configured base and mixing coefficient for the detached channel-mean target.
        if self.config.scb_gray_world_loss:
            mu_c = rgb_clear_enhanced.mean(dim=(0, 1))           # [3]
            target_c = (
                self.config.scb_gray_world_target_base
                + self.config.scb_gray_world_lambda * mu_c.detach()
            )
            scb_gray_world_loss = (
                ((mu_c - target_c) ** 2).sum() * self.config.scb_gray_world_weight
            )
        else:
            scb_gray_world_loss = torch.tensor(0.0).to(self.device)

        # Load and sample optional per-frame color-chart supervision in native image coordinates.
        chart_loss = torch.tensor(0.0, device=self.device)
        cs = self._chart_supervision
        chart_loss_active = (
            self.config.chart_supervised_loss
            and cs is not None
            and self.step > self.config.chart_supervised_warmup
            and "image_stem" in batch
        )
        if chart_loss_active and cs is not None:
            stem = batch["image_stem"]
            info = cs["frames"].get(stem)
            if info is not None:
                # Select the render path to compare with the annotated color reference.
                target_render = (
                    outputs["rgb_enhanced"]
                    if self.config.chart_supervision_target == "enhanced"
                    else rgb_clear_enhanced
                )
                H_r, W_r = target_render.shape[:2]
                H_n, W_n = info["image_size_hw"]
                sx = W_r / W_n
                sy = H_r / H_n
                patches = info["patches_native_xy"]  # list of 18 [x, y]
                r = self.config.chart_supervised_patch_radius
                samples = []
                for (x_n, y_n) in patches:
                    xc = int(round(x_n * sx))
                    yc = int(round(y_n * sy))
                    x0 = max(0, xc - r); x1 = min(W_r, xc + r + 1)
                    y0 = max(0, yc - r); y1 = min(H_r, yc + r + 1)
                    if x1 <= x0 or y1 <= y0:
                        # patch fell outside the (downscaled) frame; skip.
                        samples.append(None)
                        continue
                    # mean over the window keeps the gradient distributed
                    # to surrounding gaussians (median would discard most grad).
                    samples.append(target_render[y0:y1, x0:x1].reshape(-1, 3).mean(dim=0))
                valid = [s for s in samples if s is not None]
                if len(valid) > 0:
                    pred = torch.stack(valid, dim=0)                            # [N, 3]
                    ref_idx = [i for i, s in enumerate(samples) if s is not None]
                    ref = self._chart_ref_rgb[ref_idx].to(pred.device)          # [N, 3]
                    chart_loss = (pred - ref).abs().mean() * self.config.chart_supervised_weight
        # Reconstruction: low-light render vs raw GT.
        # Optionally compute on sRGB-encoded values to lift dark-region
        # gradients and break the SeaThru flat-σ ambiguity.
        if self.config.encode_recon_loss:
            inv_g = 1.0 / self.config.encode_gamma
            gt_for_loss = gt_img.clamp(1e-4, 1.0).pow(inv_g)
            pred_for_loss = pred_img.clamp(1e-4, 1.0).pow(inv_g)
            pred_for_loss_detach = pred_for_loss.detach()
        else:
            gt_for_loss = gt_img
            pred_for_loss = pred_img
            pred_for_loss_detach = pred_img_detach

        if self.config.water_splatting_loss:
            recon_loss = torch.abs(
                (gt_for_loss - pred_for_loss) / (pred_for_loss_detach + 1e-3)
            ).mean()
            simloss = 1 - self.ssim(
                (gt_for_loss / (pred_for_loss_detach + 1e-3)).permute(2, 0, 1)[None, ...],
                (pred_for_loss / (pred_for_loss_detach + 1e-3)).permute(2, 0, 1)[None, ...],
            )
            main_loss = (1 - self.config.ssim_lambda) * recon_loss + self.config.ssim_lambda * simloss
        else:
            Ll1 = torch.abs(gt_for_loss - pred_for_loss).mean()
            simloss = 1 - self.ssim(
                gt_for_loss.permute(2, 0, 1)[None, ...],
                pred_for_loss.permute(2, 0, 1)[None, ...],
            )
            main_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss
        loss_dict = {
            "main_loss": main_loss,
            "bs_loss": bs_loss,
            "scale_reg": scale_reg,
            "color_loss": color_loss,
            "white_balance_loss": white_balance_loss,
            "wb_anchor_loss": wb_anchor_loss,
            "clear_sparsity_loss": clear_sparsity_loss,
            "opacity_l1_loss": opacity_l1_loss,
            "alpha_decay_loss": alpha_decay_loss,
            "sigma_order_loss": sigma_order_loss,
            "scb_gray_world_loss": scb_gray_world_loss,
            "chart_supervised_loss": chart_loss,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    @torch.no_grad()
    def save_ply(self, path: str) -> None:
        """Export 3D Gaussian parameters to a standard PLY file.

        The format is compatible with common Gaussian Splatting viewers.

        Args:
            path: destination file path (should end with .ply)
        """
        import os
        from plyfile import PlyData, PlyElement

        os.makedirs(os.path.dirname(path), exist_ok=True)

        means = self.means.detach().cpu().numpy()                    # [N, 3]
        normals = np.zeros_like(means)                               # [N, 3] placeholder
        features_dc = self.features_dc.detach().cpu().numpy()        # [N, 3]
        features_rest = (
            self.features_rest.detach()
            .transpose(1, 2)                                         # [N, 3, sh-1] -> [N, 3, sh-1]
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )                                                            # [N, 3*(sh-1)]
        opacities = self.opacities.detach().cpu().numpy()            # [N, 1]
        scales = self.scales.detach().cpu().numpy()                  # [N, 3]
        quats = self.quats.detach().cpu().numpy()                    # [N, 4]

        # build attribute name list
        attr_names = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(features_dc.shape[1]):
            attr_names.append(f"f_dc_{i}")
        for i in range(features_rest.shape[1]):
            attr_names.append(f"f_rest_{i}")
        attr_names.append("opacity")
        for i in range(scales.shape[1]):
            attr_names.append(f"scale_{i}")
        for i in range(quats.shape[1]):
            attr_names.append(f"rot_{i}")

        dtype_full = [(name, "f4") for name in attr_names]
        elements = np.empty(means.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            [means, normals, features_dc, features_rest, opacities, scales, quats],
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
        CONSOLE.log(f"Saved {means.shape[0]} gaussians to {path}")

    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        if not self.training:
            with torch.no_grad():
                outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        else:
            outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        # predicted_rgb = outputs["rgb_clear"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        # Diagnostic scalars (mean of per-pixel maps) — propagated to output.json
        # via nerfstudio's get_average_image_metrics aggregation.
        metrics_dict["gaussian_count"] = float(self.num_points)
        if "alpha_coef" in outputs:
            a_coef = outputs["alpha_coef"]
            metrics_dict["alpha_coef"] = float(a_coef.mean().item())
            if a_coef.shape[-1] == 3:
                for i in range(3):
                    metrics_dict[f"alpha_coef_{i}"] = float(a_coef[..., i].mean().item())
        for i in range(3):
            if "gamma_coef" in outputs:
                metrics_dict[f"gamma_coef_{i}"] = float(outputs["gamma_coef"][..., i].mean().item())
            if "enhanced_color_rgb" in outputs:
                metrics_dict[f"enhanced_color_rgb{i}"] = float(outputs["enhanced_color_rgb"][..., i].mean().item())
        # color_rgb may be 1ch (LLNeRF L) or 3ch.
        if "color_rgb" in outputs:
            c_rgb = outputs["color_rgb"]
            for i in range(c_rgb.shape[-1]):
                metrics_dict[f"color_rgb_{i}"] = float(c_rgb[..., i].mean().item())

        images_dict = {
            "gt": output_gt_rgb,
            "rgb_medium": outputs["rgb_medium"],
            "rgb_object": outputs["rgb_object"],
            "depth": outputs["depth"],
            "rgb": outputs["rgb"],
            "rgb_clear": outputs["rgb_clear"],
            "rgb_clear_enhanced": outputs.get("rgb_clear_enhanced", outputs["rgb_clear"]),
            "rgb_enhanced": outputs.get("rgb_enhanced", outputs["rgb"]),
        }

        return metrics_dict, images_dict

    def gamma_correction_torch(self, image: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        inv_gamma = 1.0 / gamma
        image = torch.clamp(image, min=0.0, max=1.0)
        corrected_image = torch.pow(image, inv_gamma)

        if image.max() > 1.0:
            corrected_image = corrected_image * 255.0
        return corrected_image

    def compute_minkowski_norm_torch(self, image: torch.Tensor, p: float) -> torch.Tensor:
        if p == float('inf'):
            return torch.amax(image, dim=(0, 1))
        else:
            return torch.mean(torch.pow(image, p), dim=(0, 1)) ** (1.0 / p)

    def adjust_lambda_torch(self, normal_color_count: float, lambda_default: float = 0.2, lambda_max: float = 0.5) -> float:
        if normal_color_count == 0:
            return lambda_max

        lambda_val = -normal_color_count + lambda_max + lambda_default

        lambda_val = max(0.0, min(lambda_val, lambda_max))
        return lambda_val

    def shades_of_grey_white_balance_torch(self, image: torch.Tensor, p: float = 1, lambda_default: float = 0.2,
                                           gamma: float = 1.2, clamp_max: float = 7) -> torch.Tensor:
        assert not image.requires_grad, "Gradient tracking is enabled for 'image'"

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        if gamma != 1.0:
            image = self.gamma_correction_torch(image, gamma=gamma)
        mu_ref = self.compute_minkowski_norm_torch(image, p)  # 形状为 [3]
        hist_size = 32
        hist_range = [0.0, 1.0]
        bins = torch.linspace(hist_range[0], hist_range[1], hist_size + 1, device=image.device)

        histograms = []
        for c in range(3):
            channel = image[:, :, c]
            hist = torch.histc(channel, bins=hist_size, min=hist_range[0], max=hist_range[1])
            histograms.append(hist)

        color_count = sum([torch.sum(hist > 0).item() for hist in histograms])

        normal_color_count = color_count / (hist_size * 3)

        lambda_val = self.adjust_lambda_torch(normal_color_count, lambda_default=lambda_default)
        # estimated illumination μI
        mu_I = 0.5 + lambda_val * mu_ref  # μI should be in the range [0.5, 1]
        scale = (mu_I / (mu_ref + 1e-6)) * 0.7
        scale = torch.clamp(scale, min=0.1, max=clamp_max)


        balanced_image = image * scale.view(1, 1, 3)
        balanced_image = torch.clamp(balanced_image, 0.0, 1.0)
        if image.max() > 1.0:
            balanced_image = balanced_image * 255.0

        # import matplotlib.pyplot as plt
        # _image = image.detach().cpu().numpy()
        # plt.subplot(1, 2, 1)
        # plt.imshow(_image)
        # plt.title("Original Image")
        # plt.axis("off")
        #
        # _balanced_image = balanced_image.detach().cpu().numpy()
        # plt.subplot(1,2,2)
        # plt.imshow(_balanced_image)
        # plt.title("balance Image without clamp")
        # plt.axis("off")
        # plt.show()
        return balanced_image

    def simple_color_balance_tensor(self, image: torch.Tensor) -> torch.Tensor:
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        Bavg = torch.mean(b)
        Gavg = torch.mean(g)
        Ravg = torch.mean(r)
        Max = torch.max(torch.stack([Bavg, Gavg, Ravg]))
        ratio = Max / torch.stack([Bavg, Gavg, Ravg])
        satLevel = 0.005 * ratio
        imgRGB_orig = torch.stack([b.flatten(), g.flatten(), r.flatten()])
        imRGB = torch.zeros_like(imgRGB_orig)
        for ch in range(3):
            q_low = satLevel[ch].item()
            q_high = 1 - satLevel[ch].item()
            tiles = torch.quantile(imgRGB_orig[ch, :], torch.tensor([q_low, q_high], device=image.device))
            temp = torch.clamp(imgRGB_orig[ch, :], min=tiles[0], max=tiles[1])
            pmin = temp.min()
            pmax = temp.max()
            imRGB[ch, :] = (temp - pmin) * 1.0 / (pmax - pmin + 1e-8)
        H, W, _ = image.shape
        output = torch.zeros_like(image)
        output[:, :, 0] = imRGB[0, :].reshape(H, W)
        output[:, :, 1] = imRGB[1, :].reshape(H, W)
        output[:, :, 2] = imRGB[2, :].reshape(H, W)
        output = output.clamp(0.0, 1.0)
        return output

    """""------------------------------"""

    def min_dark_channel(self, img: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        dark = img.min(dim=1, keepdim=True)[0]  # [1, 1, H, W]
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=img.device)
        dark_eroded = F.conv2d(dark, kernel, padding=kernel_size // 2, groups=1)
        dark_eroded = dark_eroded / (kernel_size * kernel_size)
        dark_dilated = -F.max_pool2d(-dark_eroded, kernel_size, stride=1, padding=kernel_size // 2)
        return dark_dilated

    def mid_dark_channel(self, img: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        dark = img.min(dim=1, keepdim=True)[0]  # [1, 1, H, W]
        dark_median = kornia.filters.median_blur(dark, kernel_size=kernel_size)
        return dark_median

    def kmeans_dark_channel(self, img: torch.Tensor, kernel_size: int = 15, k: int = 8, max_iters: int = 10) -> torch.Tensor:
        B, C, H, W = img.shape
        Z = img.view(B, C, -1).permute(0, 2, 1)  # [1, H*W, 3]

        indices = torch.randperm(H * W)[:k]
        centers = Z[0, indices, :].clone()  # [k, 3]

        for _ in range(max_iters):

            distances = torch.cdist(Z, centers.unsqueeze(0), p=2)  # [1, H*W, k]

            labels = distances.argmin(dim=2)  # [1, H*W]

            new_centers = []
            for i in range(k):
                mask = (labels == i).float().unsqueeze(2)  # [1, H*W, 1]
                if mask.sum() == 0:
                    new_centers.append(centers[i])
                else:
                    new_center = (Z * mask).sum(dim=1) / mask.sum()
                    new_centers.append(new_center[0])
            centers = torch.stack(new_centers)  # [k, 3]

        res = centers[labels.squeeze(0)]  # [H*W, 3]
        res2 = res.view(B, C, H, W)  # [1, 3, H, W]

        dark = res2.min(dim=1, keepdim=True)[0]  # [1, 1, H, W]

        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=img.device)

        dark_eroded = F.conv2d(dark, kernel, padding=kernel_size // 2, groups=1)
        dark_eroded = dark_eroded / (kernel_size * kernel_size)

        return dark_eroded

    def add_dark_channels(self, dark1: torch.Tensor, dark2: torch.Tensor, beta: float = 0.7) -> torch.Tensor:
        return beta * dark1 + (1 - beta) * dark2

    def gauss_add(self, Fdark: torch.Tensor, mindark: torch.Tensor, a1: float = 4, a2: float = 2,
                  z: float = 0.6) -> torch.Tensor:
        gx = a2 * torch.exp(-((1.5 - Fdark) ** 2) / z)
        Tx = gx + a1
        Ig = (a1 * Fdark + gx * mindark) / Tx
        return Ig

    def light_channel(self, img: torch.Tensor, dark: torch.Tensor, rate: float = 0.001) -> torch.Tensor:
        B, C, H, W = img.shape
        num_pixels = max(int(H * W * rate), 1)

        flat_dark = dark.view(B, -1)  # [1, H*W]

        _, indices = torch.topk(flat_dark, num_pixels, dim=1, largest=True, sorted=False)  # [1, num_pixels]

        brightest = img.view(B, C, -1).permute(0, 2, 1).gather(1, indices.unsqueeze(-1).repeat(1, 1,
                                                                                               C))  # [1, num_pixels, 3]

        brightness = brightest.mean(dim=2)  # [1, num_pixels]

        max_brightness, max_idx = brightness.max(dim=1)  # [1], [1]
        A = brightest[0, max_idx, :].unsqueeze(0)  # [1, 3]

        return A

    def transmission_estimate(self, img: torch.Tensor, A: torch.Tensor, dark_channel: torch.Tensor, omega: float = 0.95,
                              size: int = 15) -> torch.Tensor:
        A = A.view(1, 3, 1, 1)  # [1, 3, 1, 1]

        norm_img = img / A  # [1, 3, H, W]

        min_dc = self.min_dark_channel(norm_img, size)  # [1, 1, H, W]

        transmission = 1 - omega * min_dc
        transmission = torch.clamp(transmission, 0, 1)

        return transmission

    def guided_filter(self, img: torch.Tensor, p: torch.Tensor, r: int = 60, eps: float = 1e-4) -> torch.Tensor:
        mean_guide = kornia.filters.box_blur(img, kernel_size=(r, r))
        mean_p = kornia.filters.box_blur(p, kernel_size=(r, r))
        mean_guide_p = kornia.filters.box_blur(img * p, kernel_size=(r, r))
        cov_guide_p = mean_guide_p - mean_guide * mean_p

        mean_guide_sq = kornia.filters.box_blur(img * img, kernel_size=(r, r))
        var_guide = mean_guide_sq - mean_guide * mean_guide

        a = cov_guide_p / (var_guide + eps)
        b = mean_p - a * mean_guide

        mean_a = kornia.filters.box_blur(a, kernel_size=(r, r))
        mean_b = kornia.filters.box_blur(b, kernel_size=(r, r))

        q = mean_a * img + mean_b
        return q

    def transmission_refine(self, img: torch.Tensor, transmission: torch.Tensor, r: int = 60,
                            eps: float = 1e-4) -> torch.Tensor:
        gray = kornia.color.rgb_to_grayscale(img)  # [1, 1, H, W]

        refined_transmission = self.guided_filter(gray, transmission, r=r, eps=eps)
        refined_transmission = torch.clamp(refined_transmission, 0, 1)

        return refined_transmission

    def recover_image(self, img: torch.Tensor, transmission: torch.Tensor, A: torch.Tensor, tx: float = 0.1) -> torch.Tensor:
        t = torch.clamp(transmission, min=tx)  # [1, 1, H, W]
        recovered = (img - A.view(1, 3, 1, 1)) / t + A.view(1, 3, 1, 1)
        recovered = torch.clamp(recovered, 0, 1)
        return recovered

    def bs_dcp(self, image, beta: float = 0.7, k: int = 8, rate: float = 0.001, size: int = 15):

        img = image  # [H, W, 3]
        if img.ndim == 2:
            img = img.unsqueeze(-1).repeat(1, 1, 3)  # turn to RGB
        img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        min_dc = self.min_dark_channel(img, size)  # [1, 1, H, W]
        mid_dc = self.mid_dark_channel(img, size)  # [1, 1, H, W]
        kmeans_dc = self.kmeans_dark_channel(img, size, k)  # [1, 1, H, W]

        added_dark = self.add_dark_channels(mid_dc, kmeans_dc, beta)  # [1, 1, H, W]
        combined_dark = self.gauss_add(added_dark, min_dc)  # [1, 1, H, W]

        A = self.light_channel(img, combined_dark, rate)  # [1, 3]

        transmission = self.transmission_estimate(img, A, combined_dark, omega=0.95, size=size)  # [1, 1, H, W]

        refined_transmission = self.transmission_refine(img, transmission)  # [1, 1, H, W]

        result = self.recover_image(img, refined_transmission, A, tx=0.1)  # [1, 3, H, W]
        result = result.squeeze(0).permute(1, 2, 0)
        # import matplotlib.pyplot as plt
        # _image = image.detach().cpu().numpy()
        # plt.subplot(1, 2, 1)
        # plt.imshow(_image)
        # plt.title("Original Image")
        # plt.axis("off")
        #
        # dcp_image = result.detach().cpu().numpy()
        # plt.subplot(1,2,2)
        # plt.imshow(dcp_image)
        # plt.title("dcp image")
        # plt.axis("off")
        # plt.show()

        return result

    """----------------------------------"""

    def grad(self,img):# wrong
        img = img.permute(2, 0, 1)  # reverse the order of dimensions -> [3, H, W]
        # img = img.unsqueeze(0)  # increase the batch dimension -> [1, 3, H, W]
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)[None, None, :].expand(1, 3, -1, -1).to(img.device)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32)[None, None, :].expand(1, 3, -1, -1).to(img.device)
        grad_x = F.conv2d(img[None, :], sobel_x)
        grad_y = F.conv2d(img[None, :], sobel_y)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad

    def get_grad_loss(self,img1, img2):

        g1 = self.grad(img1)
        g2 = self.grad(img2)
        scale = img1.mean() / img2.mean()
        g1 = g1
        g2 = g2 * scale
        g1 = g1.repeat(1, 3, 1, 1)  # [1, 3, H, W]
        g2 = g2.repeat(1, 3, 1, 1)  # [1, 3, H, W]
        return 0.1* (1.0 - self.ssim(g1, g2))
    """----------------------------------"""

def get_gray_loss(enhance_pooled):
    # calculate the channel standard deviation of each pixel to encourage the image to tend to grayscale
    gray_std = enhance_pooled.std(dim=1, keepdim=True)  # [B, 1, H, W]
    return gray_std.mean()

def compute_color_loss(enhance, get_gray_loss, fixed_exposure=0.8, exposure_loss_lambda=0.1, gray_loss_lambda=0.5):

    if enhance.ndim == 3:
        enhance = enhance.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    enhance_pooled = nn.functional.avg_pool2d(enhance, kernel_size=5)  # [1, 3, H//5, W//5]

    exposure = enhance_pooled.mean(dim=(2, 3))  # [1, 3]
    exposure_loss = ((exposure - fixed_exposure) ** 2).mean()  # [1]
    image_enhance_loss = exposure_loss_lambda * exposure_loss  # [1]

    gray_loss = gray_loss_lambda * get_gray_loss(enhance_pooled)  # [1]

    color_loss = gray_loss + image_enhance_loss
    return color_loss

