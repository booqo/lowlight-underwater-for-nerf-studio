"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

from .. import cudalight as _C
from ..utils.utils import bin_and_sort_gaussians, compute_cumulative_intersects


def rasterize_gaussians(
    xys: Float[Tensor, "*batch 2"],
    xys_grad_abs: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    medium_rgb: Float[Tensor, "*height width channels"],
    medium_bs: Float[Tensor, "*height width channels"],
    medium_attn: Float[Tensor, "*height width channels"],
    color_enhance: Float[Tensor, "*height width channels"], # new
    img_height: int,
    img_width: int,
    block_width: int,
    background: Optional[Float[Tensor, "*height width channels"]] = None,
    return_alpha: Optional[bool] = False,
    step: Optional[int] = None,
) -> Tensor:
    """Rasterizes a 2D Gaussian by sorting and bucketing the Gaussian intersections within each tile and returns an N-dimensional output using alpha compositing.

        NOTE:
            This function can differentiate the xys, conics, colors, and opacity inputs.

        Parameters：
            xys (Tensor): The xy coordinates of the 2D Gaussian.
            xys_grad_abs (Tensor): The absolute value of the gradient to be edited.
            depths (Tensor): The depth of the 2D Gaussian.
            radii (Tensor): The radius of the 2D Gaussian.
            conics (Tensor): The inverse covariance of the 2D Gaussian (in upper triangular format).
            num_tiles_hit (Tensor): The number of tiles hit by each Gaussian.
            colors (Tensor): The N-dimensional features associated with the Gaussian.
            opacity (Tensor): The opacity associated with the Gaussian.
            medium_rgb (Tensor): The medium RGB color.
            medium_bs (Tensor): The scattering coefficient of the medium.
            medium_attn (Tensor): The attenuation coefficient of the medium.
            img_height (int): The height of the rendered image.
            img_width (int): The width of the rendered image.
            block_width (int): Must match the block width used in the project_gaussians call.
            integer distance between pixels, 2 to 16 inclusive.
            background (Tensor): The background color.
            return_alpha (bool): Whether to return the alpha channel.
        Return Values：
            A Tensor:

            - **out_img** (Tensor): The N-dimensional render output object.
            - **out_clr** (Tensor): The N-dimensional render output clear object.
            - **out_medium** (Tensor): The N-dimensional render output medium.
            - **depth_im** (Tensor): The N-dimensional render output depth image.
            - **out_alpha** (Optional[Tensor]): The alpha channel of the render output image.
        """

    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if background is None:
        background = torch.ones(
            colors.shape[-1], dtype=torch.float32, device=colors.device
        )

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    return _RasterizeGaussians.apply(
        xys.contiguous(),
        xys_grad_abs.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        opacity.contiguous(),
        medium_rgb.contiguous(),
        medium_bs.contiguous(),
        medium_attn.contiguous(),
        color_enhance.contiguous(), # new
        img_height,
        img_width,
        block_width,
        background.contiguous(),
        return_alpha,
        step,
    )


class _RasterizeGaussians(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        xys_grad_abs: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        medium_rgb: Float[Tensor, "*height width channels"],
        medium_bs: Float[Tensor, "*height width channels"],
        medium_attn: Float[Tensor, "*height width channels"],
        color_enhance: Float[Tensor, "*height width channels"],#new
        img_height: int,
        img_width: int,
        block_width: int,
        background: Optional[Float[Tensor, "channels"]] = None,
        return_alpha: Optional[bool] = False,
        step: Optional[int] = None,
    ) -> Tensor:
        num_points = xys.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
                * background
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, device=xys.device)
            first_idx = torch.zeros(img_height, img_width, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )
            if colors.shape[-1] == 3: # 开始渲染
                rasterize_fn = _C.rasterize_forward
            else:
                rasterize_fn = _C.nd_rasterize_forward

            out_img, out_clr, out_medium, depth_im, final_Ts, final_idx, first_idx = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                opacity,
                medium_rgb,
                medium_bs,
                medium_attn,
                color_enhance, # new
                depths,
                background,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.num_intersects = num_intersects
        ctx.block_width = block_width
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            xys_grad_abs,
            conics,
            colors,
            opacity,
            medium_rgb,
            medium_bs,
            medium_attn,
            color_enhance, # new
            depths,
            background,
            final_Ts,
            final_idx,
            first_idx,
        )
        
        if return_alpha:
            out_alpha = 1 - final_Ts
            return out_img, out_clr, out_medium, depth_im, out_alpha
        else:
            return out_img, out_clr, out_medium, depth_im

    @staticmethod
    def backward(ctx, v_out_img, v_out_clr, v_out_medium, v_depth_im, v_out_alpha=None):
        img_height = ctx.img_height
        img_width = ctx.img_width
        num_intersects = ctx.num_intersects

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            xys_grad_abs,
            conics,
            colors,
            opacity,
            medium_rgb,
            medium_bs,
            medium_attn,
            color_enhance, # new
            depths,
            background,
            final_Ts,
            final_idx,
            first_idx,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)
            v_medium_rgb = torch.zeros_like(medium_rgb)
            v_medium_bs = torch.zeros_like(medium_bs)
            v_medium_attn = torch.zeros_like(medium_attn)
            v_color_enhance = torch.zeros_like(color_enhance) # new

        else:
            if colors.shape[-1] == 3:
                rasterize_fn = _C.rasterize_backward # RGB 渲染的反向传播
            else:
                rasterize_fn = _C.nd_rasterize_backward  # 多通道渲染的反向传播
            # 计算梯度
            v_xy, v_conic, v_colors, v_opacity, v_medium_rgb, v_medium_bs, v_medium_attn, v_color_enhance = rasterize_fn(
                img_height,
                img_width,
                ctx.block_width,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                xys_grad_abs,
                conics,
                colors,
                opacity,
                medium_rgb,
                medium_bs,
                medium_attn,
                color_enhance, # new
                depths,
                background,
                final_Ts,
                final_idx,
                first_idx,
                v_out_img,
                v_out_clr,
                v_out_medium,
                v_out_alpha,
            )
            
        return (
            v_xy,  # xys
            None,  # xys_grad_abs
            None,  # depths
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            v_opacity,  # opacity
            v_medium_rgb,  # medium_rgb
            v_medium_bs,  # medium_bs
            v_medium_attn,  # medium_attn
            v_color_enhance, # new
            None,  # img_height
            None,  # img_width
            None,  # block_width
            None,  # background
            None,  # return_alpha
            None,  # step
        )
