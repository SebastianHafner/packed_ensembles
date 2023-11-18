import torch
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod
import math
import numpy as np
from utils import augmentations, helpers
import cv2
from affine import Affine


class AbstractPopulationDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.patch_size = cfg.DATALOADER.PATCH_SIZE
        self.modality = cfg.DATALOADER.MODALITY

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @staticmethod
    def pop_log_conversion(pop: float) -> float:
        if pop == 0:
            return 0
        else:
            return math.log10(pop)


# dataset for urban extraction with building footprints
class GridPopulationDataset(AbstractPopulationDataset):

    def __init__(self, cfg, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.run_type = run_type
        self.no_augmentations = no_augmentations

        self.samples = helpers.load_json(self.root_path / 'samples.json')
        self.samples = [s for s in self.samples if s['split'] == run_type]

        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg.AUGMENTATION)

        if self.modality == 's2':
            self.img, self.geotransform, self.crs = helpers.read_tif(self.root_path / 's2_mosaic_2019_10m.tif')
            self.img = np.clip(self.img / 4_000, 0, 1).astype(np.float32)
        elif self.modality == 'bf':
            self.img, self.geotransform, self.crs = helpers.read_tif(self.root_path / 'google_building_footprints.tif')
        else:
            raise Exception('Unkown modality')

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        i, j = sample['i_grid'], sample['j_grid']

        pop = float(sample['hrsl_pop'])
        pop = self.pop_log_conversion(pop) if self.cfg.DATALOADER.LOG_POP else pop

        patch = self.load_patch(self.modality, i, j)

        x = self.transform(patch)

        item = {
            'x': x,
            'y': torch.tensor([pop]),
            'i': i,
            'j': j,
        }

        return item

    def load_patch(self, modality: str, i: int, j: int) -> np.ndarray:
        if modality == 's2':
            i_s2, j_s2 = i * 10, j * 10
            patch = self.img[i_s2:i_s2 + 10, j_s2:j_s2 + 10]
        elif modality == 'bf':
            i_bf, j_bf = i * 200, j * 200
            patch = self.img[i_bf:i_bf + 200, j_bf:j_bf + 200]
            patch = patch.astype(np.float32)
        else:
            raise Exception('Unknown modality')

        # resampling images to desired patch size
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        return patch

    def get_geotransform(self, res: int):
        _, _, x_origin, _, _, y_origin, *_ = self.geotransform
        geotransform = (x_origin, res, 0.0, y_origin, 0.0, -res)
        geotransform = Affine.from_gdal(*geotransform)
        return geotransform

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
