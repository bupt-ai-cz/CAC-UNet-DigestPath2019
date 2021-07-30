import numpy as np
from .dataset import BaseDataset
# from ..aug.segaug import crop_aug, seg_aug
from ..aug.segaug import seg_aug
from ...models.registry import DATASET


@DATASET.register("DigestDataset")
class DigestDataset(BaseDataset):

    # overwrite
    def aug(self, image, label):
        return seg_aug(image, label)
