import math
import os
from pathlib import Path

import detectron2.data.transforms as DT
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from detectron2.data import detection_utils as d2_utils
from detectron2.structures import Instances, BitMasks
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.data import read_flow


class FlowEvalDetectron(Dataset):
    def __init__(self, data_dir, resolution, pair_list, val_seq, to_rgb=False, with_rgb=False, size_divisibility=None,
                 small_val=0, flow_clip=1., norm=True, read_big=True, eval_size=True, force1080p=False):
        self.val_seq = val_seq
        self.to_rgb = to_rgb
        self.with_rgb = with_rgb
        self.data_dir = data_dir
        self.pair_list = pair_list
        self.resolution = resolution

        self.eval_size = eval_size

        self.samples = []
        self.samples_fid = {}
        for v in self.val_seq:
            seq_dir = Path(self.data_dir[0]) / v
            frames_paths = sorted(seq_dir.glob('*.flo'))
            self.samples_fid[str(seq_dir)] = {fp: i for i, fp in enumerate(frames_paths)}
            self.samples.extend(frames_paths)
        self.samples = [os.path.join(x.parent.name, x.name) for x in self.samples]
        if small_val > 0:
            _, self.samples = train_test_split(self.samples, test_size=small_val, random_state=42)
        self.gaps = ['gap{}'.format(i) for i in pair_list]
        self.neg_gaps = ['gap{}'.format(-i) for i in pair_list]
        self.size_divisibility = size_divisibility
        self.ignore_label = -1
        self.transforms = DT.AugmentationList([
            DT.Resize(self.resolution, interp=Image.BICUBIC),
        ])
        self.flow_clip=flow_clip
        self.norm_flow=norm
        self.read_big=read_big
        self.force1080p_transforms=None
        if force1080p:
            self.force1080p_transforms = DT.AugmentationList([
            DT.Resize((1088, 1920), interp=Image.BICUBIC),
        ])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_dicts = []

        dataset_dict = {}
        flow_dir = Path(self.data_dir[0]) / self.samples[idx]
        fid = self.samples_fid[str(flow_dir.parent)][flow_dir]
        flo = einops.rearrange(read_flow(str(flow_dir), self.resolution, self.to_rgb), 'c h w -> h w c')
        dataset_dict["gap"] = 'gap1'

        suffix = '.png' if 'CLEVR' in self.samples[idx] else '.jpg'
        rgb_dir = (self.data_dir[1] / self.samples[idx]).with_suffix(suffix)
        gt_dir = (self.data_dir[2] / self.samples[idx]).with_suffix('.png')

        rgb = d2_utils.read_image(str(rgb_dir)).astype(np.float32)
        original_rgb = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0., 255.))).float()
        if self.read_big:
            rgb_big = d2_utils.read_image(str(rgb_dir).replace('480p', '1080p')).astype(np.float32)
            rgb_big = (torch.as_tensor(np.ascontiguousarray(rgb_big))[:, :, :3]).permute(2, 0, 1).clamp(0., 255.)
            if self.force1080p_transforms is not None:
                rgb_big = F.interpolate(rgb_big[None], size=(1080, 1920), mode='bicubic').clamp(0., 255.)[0]

        input = DT.AugInput(rgb)

        # Apply the augmentation:
        preprocessing_transforms = self.transforms(input)  # type: DT.Transform
        rgb = input.image
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0., 255.)
        d2_utils.check_image_size(dataset_dict, flo)

        if gt_dir.exists():
            sem_seg_gt_ori = d2_utils.read_image(gt_dir)
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt_ori)
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
                sem_seg_gt_ori = sem_seg_gt_ori[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
                sem_seg_gt_ori = (sem_seg_gt_ori > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))
            sem_seg_gt_ori = np.zeros((original_rgb.shape[-2], original_rgb.shape[-1]))

        gwm_dir = (Path(str(self.data_dir[2]).replace('Annotations', 'gwm')) / self.samples[idx]).with_suffix(
            '.png')
        if gwm_dir.exists():
            gwm_seg_gt = preprocessing_transforms.apply_segmentation(d2_utils.read_image(str(gwm_dir)))
            gwm_seg_gt = np.array(gwm_seg_gt)
            if gwm_seg_gt.ndim == 3:
                gwm_seg_gt = gwm_seg_gt[:, :, 0]
            if gwm_seg_gt.max() == 255:
                gwm_seg_gt[gwm_seg_gt == 255] = 1
        else:
            gwm_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Pad image and segmentation label here!
        if self.to_rgb:
            flo = torch.as_tensor(np.ascontiguousarray(flo.transpose(2, 0, 1))) / 2 + .5
            flo = flo * 255
        else:
            flo = torch.as_tensor(np.ascontiguousarray(flo.transpose(2, 0, 1)))
            if self.norm_flow:
                flo = flo/(flo ** 2).sum(0).max().sqrt()
            flo = flo.clip(-self.flow_clip, self.flow_clip)

        rgb = torch.as_tensor(np.ascontiguousarray(rgb)).float()
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            sem_seg_gt_ori = torch.as_tensor(sem_seg_gt_ori.astype("long"))
        if gwm_seg_gt is not None:
            gwm_seg_gt = torch.as_tensor(gwm_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (flo.shape[-2], flo.shape[-1])
            padding_size = [
                0,
                int(self.size_divisibility * math.ceil(image_size[1] // self.size_divisibility)) - image_size[1],
                0,
                int(self.size_divisibility * math.ceil(image_size[0] // self.size_divisibility)) - image_size[0],
            ]
            flo = F.pad(flo, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            if gwm_seg_gt is not None:
                gwm_seg_gt = F.pad(gwm_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (flo.shape[-2], flo.shape[-1])  # h, w
        if self.eval_size:
            image_shape = (sem_seg_gt_ori.shape[-2], sem_seg_gt_ori.shape[-1])


        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo
        dataset_dict["rgb"] = rgb


        dataset_dict["original_rgb"] = F.interpolate(original_rgb[None], mode='bicubic', size=sem_seg_gt_ori.shape[-2:], align_corners=False).clip(0.,255.)[0]
        if self.read_big:
            dataset_dict["RGB_BIG"] = rgb_big

        dataset_dict["category"] = str(gt_dir).split('/')[-2:]
        dataset_dict['frame_id'] = fid

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            dataset_dict["sem_seg_ori"] = sem_seg_gt_ori.long()

        if gwm_seg_gt is not None:
            dataset_dict["gwm_seg"] = gwm_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
            dataset_dicts.append(dataset_dict)

        return dataset_dicts
