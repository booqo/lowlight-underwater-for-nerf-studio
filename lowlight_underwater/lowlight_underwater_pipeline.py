"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from lowlight_underwater.lowlight_underwater_datamanager import LowlightUnderwaterConfig
from lowlight_underwater.lowlight_underwater_model import LowlightUnderwaterModel, LowlightUnderwaterModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler


@dataclass
class TemplatePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: TemplatePipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=LowlightUnderwaterConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=LowlightUnderwaterModelConfig)
    """specifies the model config"""


class TemplatePipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: TemplatePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        # Build image_idx -> filename stem table, used by chart_supervised_loss
        # to look up per-frame DGK patch coordinates. Source: train_dataset
        # _dataparser_outputs.image_filenames (ordered the same as cam_idx).
        train_filenames = self.datamanager.train_dataset._dataparser_outputs.image_filenames
        self._train_image_stems: list[str] = [Path(p).stem for p in train_filenames]

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                LowlightUnderwaterModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """Same flow as VanillaPipeline.get_train_loss_dict but stamps the
        current frame's image_idx and stem into the batch dict so model-side
        chart_supervised_loss can index its per-frame patch table without
        needing a back-reference to the datamanager."""
        camera_or_bundle, batch = self.datamanager.next_train(step)
        # FullImageDatamanager attaches cam_idx via camera.metadata; pixel-
        # sampler datamanagers don't, but those aren't used in current configs.
        meta = getattr(camera_or_bundle, "metadata", None)
        if isinstance(meta, dict) and "cam_idx" in meta:
            idx = int(meta["cam_idx"])
            batch["image_idx"] = idx
            if 0 <= idx < len(self._train_image_stems):
                batch["image_stem"] = self._train_image_stems[idx]
        model_outputs = self._model(camera_or_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict
