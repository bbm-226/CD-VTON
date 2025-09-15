import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

class OpenImageDataset(data.Dataset):
    def __init__(self, state, dataset_dir, type="paired"):
        self.state = state
        self.dataset_dir = dataset_dir

        if state == "train":
            self.dataset_file = os.path.join(dataset_dir, "train_pairs.txt")
        if state == "test":
            if type == "unpaired":
                self.dataset_file = os.path.join(dataset_dir, "test_pairs_unpaired.txt")
            if type == "paired":
                self.dataset_file = os.path.join(dataset_dir, "test_pairs_paired.txt")

        self.people_list = []
        self.clothes_list = []
        with open(self.dataset_file, 'r') as f:
            for line in f.readlines():
                people, clothes, category = line.strip().split()
                if category == "0":
                    category = "upper_body"
                elif category == "1":
                    category = "lower_body"
                elif category == "2":
                    category = "dresses"
                people_path = os.path.join(self.dataset_dir, category, "images", people)
                clothes_path = os.path.join(self.dataset_dir, category, "images", clothes)
                self.people_list.append(people_path)
                self.clothes_list.append(clothes_path)

        
    def __len__(self):
        return len(self.people_list)

    def __getitem__(self, index):
        people_path = self.people_list[index]
        clothes_path = self.clothes_list[index]
        dense_path = people_path.replace("images", "dense")[:-5] + "5_uv.npz"
        mask_path = people_path.replace("images", "mask")[:-3] + "png"
        category = people_path.split("/")[-3]
        
        img = Image.open(people_path).convert("RGB").resize((512, 512))
        img = torchvision.transforms.ToTensor()(img)
        refernce = Image.open(clothes_path).convert("RGB").resize((224, 224))
        refernce = torchvision.transforms.ToTensor()(refernce)
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        mask = torchvision.transforms.ToTensor()(mask)
        mask = 1-mask
        densepose = np.load(dense_path)
        densepose = torch.from_numpy(densepose['uv'])
        densepose = torch.nn.functional.interpolate(densepose.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True).squeeze(0)

        img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(refernce)

        inpaint = img * mask
        hint = torchvision.transforms.Resize((512, 512))(refernce)
        hint = torch.cat((hint,densepose),dim = 0)


        return {"GT": img, 
                "inpaint_image": inpaint,
                "inpaint_mask": mask, 
                "ref_imgs": refernce, 
                "hint": hint, 
                "img_name": os.path.basename(people_path),
                "cloth_name": os.path.basename(clothes_path),
                "category": category
                }


        