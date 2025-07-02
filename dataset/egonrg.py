# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
import pdb

import cv2
import numpy as np
from PIL import Image

import torch
from thirdparty.PIDNet.dataset.base_dataset import BaseDataset

class EgoNRGDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=3,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=640, 
                 crop_size=(640, 480),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(EgoNRGDataset, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)
        
        self.id2label = {
            0:'background', 
            1:'left_limb', 
            2:'right_limb'}
        
        # Define your color mapping
        self.color_to_class = {
            (0, 0, 0): 0,      # background (black)
            (0, 0, 255): 1,    # left_limb (blue)
            (0, 255, 0): 2,    # right_limb (green)
        }


        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {0:0, 1:0, 2:1}

        # Assign a weight to each class based on the number of samples in the dataset (less left limbs (id=1) than right limbs (id=2))
        self.class_weights = torch.FloatTensor([0.8000, 1.1000, 0.8500]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape
        
        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name
        
         # Convert palletized mask to class indices mask
        class_mask = self.palette_to_class_mask(os.path.join(self.root, item["label"]), self.color_to_class)
        
        # Convert mask to PIL Image for processor
        label = np.array(Image.fromarray(class_mask, mode='L'))
        
        # Convert label to class indices
        label = self.convert_label(label)

        # Generate sample for training
        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)
        
        # # Save debug images if needed
        # # Save image
        # cv2.imwrite(f'{name}_image.png', image.transpose(1,2,0)*255)
        # # Save label
        # cv2.imwrite(f'{name}_label.png', label)
        # # Save edge
        # cv2.imwrite(f'{name}_edge.png', edge*255)
        
        # pdb.set_trace()

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
    def palette_to_class_mask(self, mask_path, color_to_class_mapping):
        """Convert RGB palettized mask to class index mask"""
        mask = Image.open(mask_path).convert('RGB')
        mask_array = np.array(mask)
        
        h, w = mask_array.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        for color, class_id in color_to_class_mapping.items():
            color_match = np.all(mask_array == color, axis=2)
            class_mask[color_match] = class_id
        
        return class_mask
        
if __name__ == "__main__":
    dataset = EgoNRGDataset(
        root="/root/datasets/EgoNRG/pidnet/",
        list_path="list/egonrg/train.lst",
        num_classes=3,
        multi_scale=True,
        flip=True,
        ignore_label=255,
    )
    dataset.__getitem__(67)