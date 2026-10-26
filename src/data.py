from __future__ import annotations
#TODO : the format used should be DICOM WSI (DSI) 
"""
from wsistream.backends import TiffSlideBackend
from wsistream.datasets import TCGAAdapter
from wsistream.filters import HSVPatchFilter
from wsistream.sampling import MultiMagnificationSampler
from wsistream.tissue import CLAMTissueDetector
from wsistream.torch import WsiStreamDataset
from wsistream.transforms import ComposeTransforms, HEDColorAugmentation, ResizeTransform
from wsistream.views import ViewConfig
"""
from config import DataConfig

TARGET_SIZE = 224
TARGET_MPPS = [0.25, 0.5, 1.0, 2.0]

TILE_VIEW = "tile"

def _hed_resize(sigma: float) -> ComposeTransforms:
    return ComposeTransforms(
        transforms=[
            HEDColorAugmentation(sigma=sigma),
            ResizeTransform(target_size=TARGET_SIZE)
            ]
        )

def _resize() -> ResizeTransform:
    return ResizeTransform(target_size=TARGET_SIZE)

def _train_views(cfg: DataConfig) -> list[ViewConfig]:
    """One HED-augmented tile per sample, fed to both the teacher and the student."""
    return [ViewConfig(name=TILE_VIEW, transforms=_hed_resize(cfg.hed_sigma))]

def _common_kwargs(cfg: DataConfig, num_patches: int):
    return dict(
        backend=TiffSlideBackend(),
        tissue_detector=CLAMTissueDetector(use_otsu=True, a_t=16, a_h=4),
        sampler=MultiMagnificationSampler(
            target_mpps=TARGET_MPPS,
            patch_size=cfg.patch_size,
            num_patches=num_patches,
            tissue_threshold=cfg.tissue_threshold
            ),
        patch_filter=HSVPatchFilter(
            hue_range=(90, 180),
            sat_range=(8, 255),
            val_range=(103, 255),
            min_pixel_fraction=0.6
            ),
        dataset_adapter=TCGAAdapter()
        )

def build_train_dataset(slide_paths, cfg: DataConfig, seed: int) -> WsiStreamDataset:
    return WsiStreamDataset(
        slide_paths=slide_paths,
        views=_train_views(cfg),
        slide_sampling="random",
        pool_size=cfg.pool_size,
        patches_per_slide=cfg.patches_per_slide,
        patches_per_visit=cfg.patches_per_visit,
        cycle=True,
        seed=seed,
        **_common_kwargs(cfg, num_patches=-1)
        )

def build_val_dataset(slide_paths, cfg: DataConfig, seed: int, num_slides: int) -> WsiStreamDataset:
    return WsiStreamDataset(
        slide_paths=slide_paths,
        views=[ViewConfig(name=TILE_VIEW, transforms=_resize())],
        slide_sampling="sequential",
        pool_size=min(num_slides, cfg.pool_size),
        patches_per_slide=cfg.val_patches_per_slide,
        cycle=False,
        seed=seed,
        **_common_kwargs(cfg, num_patches=cfg.val_patches_per_slide)
        )