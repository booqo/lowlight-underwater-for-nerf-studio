"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from lowlight_underwater.lowlight_underwater_datamanager import (
    LowlightUnderwaterConfig,
)
from lowlight_underwater.lowlight_underwater_model import LowlightUnderwaterModelConfig
from lowlight_underwater.lowlight_underwater_pipeline import (
    TemplatePipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig



NUM_STEPS = 15000
lowlight_underwater_method = MethodSpecification(
    config=TrainerConfig(
        method_name="lowlight_underwater",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=NUM_STEPS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=LowlightUnderwaterModelConfig(
                num_steps=NUM_STEPS,
                zero_medium=False,
                one_color = False,
                enhance_enable = True,
                bs_loss_enable = False,
                gray_loss_enable = True,
                water_splatting_loss = True,
                white_balance_loss = False,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=NUM_STEPS,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0025,
                    max_steps=NUM_STEPS,
                ),
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0025 / 20,
                    max_steps=NUM_STEPS,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.05,
                    max_steps=NUM_STEPS,
                ),
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.005,
                    max_steps=NUM_STEPS,
                ),
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                      "scheduler": ExponentialDecaySchedulerConfig(
                          lr_final=0.001,
                          max_steps=NUM_STEPS,
                      ),
                      },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=NUM_STEPS),
            },
            "medium_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=NUM_STEPS,
                ),
            },
            "color_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=NUM_STEPS,
                ),
            },
            "enhance_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=NUM_STEPS,
                ),
            },
            "direction_encoding": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=NUM_STEPS,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Splatting for lowlight underwater scenes.",
)

lowlight_underwater_big_method = MethodSpecification(
    config = TrainerConfig(
        method_name="lowlight_underwater_big",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=15000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=LowlightUnderwaterModelConfig(
                cull_alpha_thresh=0.005,
                densify_grad_thresh=0.0005,
                continue_cull_post_densification = True,
                num_steps=40000,
                zero_medium=False,
                one_color=False,
                enhance_enable=True,
                bs_loss_enable=True,
                gray_loss_enable=True,
                water_splatting_loss=True,
                white_balance_loss = False
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=40000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0025,
                    max_steps=40000,
                ),
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0025 / 20,
                    max_steps=40000,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.05,
                    max_steps=40000,
                ),
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.005,
                    max_steps=40000,
                ),
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=40000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
                        "medium_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=40000,
                ),
            },
            "color_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=40000,
                ),
            },
            "enhance_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=40000,
                ),
            },
            "direction_encoding": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=40000,
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=40000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Splatting big for lowlight underwater scenes.",
)