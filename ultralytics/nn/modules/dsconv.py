"""Dynamic Snake Convolution for CMDrill-YOLOv12.

Adapted from DSCNet (Qi et al., ICCV 2023) — https://github.com/YaoleiQi/DSCNet
Upstream file: DSCNet_2D_opensource/Code/DRIVE/S3_DSConv_pro.py (class DSConv_pro)

Adaptation notes:
- Device is inferred from the input tensor at forward time (not hard-coded at __init__)
- Input/output tensors strictly follow (B, C, H, W) PyTorch convention
- out_channels must be divisible by 4 (GroupNorm constraint)
"""

import torch
import torch.nn as nn
import einops


class DSConv(nn.Module):
    """Dynamic Snake Convolution with learnable offset along one axis.

    Args:
        in_channels: input channel count.
        out_channels: output channel count (must be divisible by 4).
        kernel_size: number of sampling points along the snake. 9 recommended for long tubular targets.
        extend_scope: range to expand offsets, default 1.0.
        morph: 0 for horizontal snake (along x-axis), 1 for vertical snake (along y-axis).
        if_offset: whether to apply learned deformation. False degrades to standard conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
    ):
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")
        assert out_channels % 4 == 0, "out_channels must be divisible by 4 for GroupNorm."

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        offset = self.offset_conv(x)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        y_coord, x_coord = _get_coordinate_map_2d(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=device,
        )
        deformed = _get_interpolated_feature(x, y_coord, x_coord)

        if self.morph == 0:
            out = self.dsc_conv_x(deformed)
        else:
            out = self.dsc_conv_y(deformed)

        out = self.gn(out)
        out = self.relu(out)
        return out


def _get_coordinate_map_2d(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float,
    device: torch.device,
):
    """Compute 2D coordinate maps for snake sampling. Note: DSCNet upstream uses (width, height)
    naming for the last two spatial dims; we keep that for fidelity, but the PyTorch layout is still (B, C, H, W)."""
    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()
        y_offset_new_[center] = 0
        for index in range(1, center + 1):
            y_offset_new_[center + index] = y_offset_new_[center + index - 1] + y_offset_[center + index]
            y_offset_new_[center - index] = y_offset_new_[center - index + 1] + y_offset_[center - index]
        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")
    else:
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()
        x_offset_new_[center] = 0
        for index in range(1, center + 1):
            x_offset_new_[center + index] = x_offset_new_[center + index - 1] + x_offset_[center + index]
            x_offset_new_[center - index] = x_offset_new_[center - index + 1] + x_offset_[center - index]
        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def _get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
) -> torch.Tensor:
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coord = _scale_coord(y_coordinate_map, origin=(0, y_max))
    x_coord = _scale_coord(x_coordinate_map, origin=(0, x_max))

    y_coord = torch.unsqueeze(y_coord, dim=-1)
    x_coord = torch.unsqueeze(x_coord, dim=-1)

    grid = torch.cat([x_coord, y_coord], dim=-1)

    return nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )


def _scale_coord(coord: torch.Tensor, origin, target=(-1.0, 1.0)) -> torch.Tensor:
    lo, hi = origin
    a, b = target
    coord = torch.clamp(coord, lo, hi)
    scale = (b - a) / (hi - lo)
    return a + scale * (coord - lo)
