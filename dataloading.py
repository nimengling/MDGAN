import os
import tifffile
import torch
import warnings
import numpy as np
from skimage import color
# from PIL import Image
# from torchvision import transforms as TR
import torchvision.transforms.functional as F
from functions import move_to_gpu,norm
from .recommended_config import get_recommended_config

def prepare_dataloading(opt):
    dataset = Dataset(opt)
    recommended_config = {"image resolution": dataset.image_resolution,
                          "noise_shape": dataset.recommended_config[0], 
                          "num_blocks_g":  dataset.recommended_config[1],
                          "num_blocks_d":  dataset.recommended_config[2],
                          "num_blocks_d0": dataset.recommended_config[3],
                          "no_masks": dataset.no_masks,
                          "num_mask_channels": dataset.num_mask_channels}
    if not recommended_config["no_masks"] and not opt.no_masks:
        print("Using the training regime *with* segmentation masks")
    else:
        opt.no_masks = True
        print("Using the training regime *without* segmentation masks")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = True, num_workers=0)
    return dataloader, recommended_config


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        """
        The dataset class. Supports both regimes *with* and *without* segmentation masks.
        """
        self.device = opt.device
        # --- images --- #
        self.root_images = os.path.join(opt.dataroot, opt.dataset_name, "image")
        self.root_masks = os.path.join(opt.dataroot, opt.dataset_name, "mask")
        self.list_imgs = self.get_frames_list(self.root_images)
        assert len(self.list_imgs) > 0, "Found no images"
        self.image_resolution, self.recommended_config = get_recommended_config(self.get_im_resolution(opt.max_size))
        self.nc_im = opt.nc_im

        # --- masks --- #
        if os.path.isdir(self.root_masks) and not opt.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        else:
            self.no_masks = True
            self.num_mask_channels = None

        print("Created a dataset of size =", len(self.list_imgs), "with image resolution", self.image_resolution)

    def get_frames_list(self, path):
        return sorted(os.listdir(path))

    def __len__(self):
        return 100000000  # so first epoch finishes only with break

    def get_im_resolution(self, max_size):
        """
        Iterate over images to determine image resolution.
        If there are images with different shapes, return the square of average size
        """
        res_list = list()
        for cur_img in self.list_imgs:
            img_pil = tifffile.imread(os.path.join(self.root_images, cur_img))
            if len(img_pil.shape) == 4:
                img_shape = (img_pil.shape[0], img_pil.shape[1], img_pil.shape[2])
                res_list.append(img_shape)
            else:
                res_list.append(img_pil.shape)

        all_res_equal = len(set(res_list)) <= 1  
        if all_res_equal:
            size_1, size_2, size_3 = res_list[0]  # all images have same resolution -> using original resolution
        else:
            warnings.warn("Images in the dataset have different resolutions. Resizing them to squares of mean size.")
            size_1 = size_2 = size_3 = sum([sum(item) for item in res_list]) / (2 * len(res_list))
        size_1, size_2, size_3 = self.bound_resolution(size_1, size_2, size_3,max_size)
        return size_3, size_2, size_1

    def bound_resolution(self, size_1, size_2, size_3, max_size):
        """
        Ensure the image shape does not exceed --max_size
        """
        if size_1 > max_size:
            size_1, size_2, size_3 = max_size, size_2 / (size_1 / max_size), size_3 / (size_1 / max_size)
        if size_2 > max_size:
            size_1, size_2, size_3 = size_1 / (size_2 / max_size), max_size, size_3 / (size_2 / max_size)
        if size_3 > max_size:
            size_1, size_2, size_3 = size_1 / (size_3 / max_size), size_2 / (size_3 / max_size), max_size
        return int(size_1), int(size_2), int(size_3)


    def __getitem__(self, index):
        output = dict()
        idx = index % len(self.list_imgs) #idx=0
        target_size = self.image_resolution

        # --- image ---#
        x = tifffile.imread(os.path.join(self.root_images, self.list_imgs[idx]))
        if x.shape == (x.shape[0], x.shape[1], x.shape[2]):
            x = np.stack((x,) * 3, axis=-1)   
        if self.nc_im == 3:
            x = x.transpose(( 3, 0, 1, 2)) / 255  
        img = torch.from_numpy(x)
        img = img.type(torch.cuda.FloatTensor)
        img = norm(img)
        output["images"] = img  
     

        # --- mask ---#
        if not self.no_masks:
            raise NotImplementedError("w/o --no_masks is not implemented in this release")
        return output



