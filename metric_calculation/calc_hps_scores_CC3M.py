from PIL import Image
import os
device = 'cuda:0'
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import warnings
from pathlib import Path
import pandas as pd
import os

import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torchvision
import numpy as np
from datasets import load_dataset
warnings.filterwarnings("ignore")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# import
import hpsv2
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

# load model
device = "cuda"
pickscore_processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
pickscore_model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

pickscore_processor = AutoProcessor.from_pretrained(pickscore_processor_name_or_path)
pickscore_model = AutoModel.from_pretrained(pickscore_model_pretrained_name_or_path).eval().to(device)

def pickscore(prompt, images):
    
    # preprocess
    image_inputs = pickscore_processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = pickscore_processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = pickscore_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = pickscore_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        #probs = torch.softmax(scores, dim=-1)
    
    return scores.cpu().tolist()

import ImageReward as RM
model_imagereward = RM.load("ImageReward-v1.0")

res_op = torchvision.transforms.Resize(size=(512,512))
def collate_fn(data):
    images_torch = torch.stack([res_op(torchvision.transforms.ToTensor()(Image.fromarray(np.array(example["jpg"])).convert('RGB'))) for example in data])
    images_pil = [Image.fromarray(np.array(example["jpg"])).convert('RGB') for example in data]
    img_names = [example["__key__"] for example in data]
    img_txt = [example["txt"] for example in data]

    return {
        "images_torch": images_torch,
        "images_pil": images_pil,
        "img_names": img_names,
        "img_txt": img_txt
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to calculate HPSv2, ImageReward and PickScore values on CC3M dataset.")
    parser.add_argument(
        "--range_low",
        type=int,
        default=None,
        required=True,
        help="start index of the dataset (included)",
    )
    parser.add_argument(
        "--range_high",
        type=int,
        default=None,
        required=True,
        help="end index of the dataset (excluded)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--ft_save_path",
        type=str,
        required=True,
        help="Path to save feather file",
    )
    args = parser.parse_args()
    # used to parallelize computations by running multiple instances of script on different parts of the dataset simultaneously
    range_low = int(args.range_low)
    range_high = int(args.range_high)
    print("RANGE: ",range_low, " ",range_high)
    
    ds = load_dataset("pixparse/cc3m-wds", cache_dir='/test/datasets/cc3m', # replace with your path to the dataset
                       split=f'train[{range_low}:{range_high}]')
    train_dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    df = pd.DataFrame(columns=['image_file', 'text'] + ['image_reward', 'pickscore', 'hpsv2'])

    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        imgs_pil = data['images_pil']
        imgs_torch = data['images_torch']
        texts = data['img_txt']
        img_names = data['img_names']
        outputs = {}
        #print(imgs_pil[0].mode)
        #print(imgs_torch.shape)
        for i in range(len(img_names)):
            img = imgs_pil[i]
            prompt = texts[i]
            row = {'image_file':img_names[i], 'text':prompt}
            with torch.no_grad():
                try:
                    reward_score = model_imagereward.score(prompt, [img])
                except:
                    reward_score = np.nan
                row['image_reward'] = reward_score
                try:
                    row['pickscore'] = pickscore(prompt, [img])[0]
                except:
                    row['pickscore'] = np.nan
                try:
                    row['hpsv2'] = hpsv2.score(img, prompt, hps_version="v2.0") [0]
                except:
                    row['hpsv2'] = np.nan
            df.loc[len(df)] = row
            
        #print(df)

    df.to_feather(args.ft_save_path)
