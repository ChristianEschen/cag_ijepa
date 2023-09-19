
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.utils.misc import first
import random
import numpy as np
import torch
from monai.transforms import (Compose)
from monai.transforms import EnsureChannelFirstD, AddChanneld, RepeatChanneld, SpatialPadd

from dictionary import LoadImaged
import highdicom
from highdicom.io import ImageFileReader
def load_random_frame(filename_or_obj):
    with ImageFileReader(filename_or_obj) as image:
        frame_idx = random.randint(0, image.number_of_frames - 1)
        frame = image.read_frame(frame_idx)
    return frame
    
# define this file as main
if __name__ == '__main__':
    path1 = "/home/alatar/miacag/data/angio/sample_data/2/0002.dcm"
    path2 = "/home/alatar/miacag/data/angio/sample_data/6/0009.DCM"
    images = [
        path1,
        path2
    ]

    labels = np.array([0, 1], dtype=np.int64)
    train_files = [{"img": img, "label": label} for img, label in zip(images, labels)]
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="HighdicomReader", image_only=True),
            AddChanneld(keys=["img"]),
            # SpatialPadd(
            #         keys=["img"],
            #         spatial_size=[548,
            #                     548]),
            RepeatChanneld(keys=["img"], repeats=3),
        ]
    )
    # Define dataset, data loader
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    print(train_transforms(train_files)[0]['img'].shape)
   # check_data = first(check_loader)
    print(check_data["img"].shape, check_data["label"])