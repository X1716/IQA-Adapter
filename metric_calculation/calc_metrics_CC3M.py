from PIL import Image
import os
device = 'cuda:0'
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import warnings
from pathlib import Path
import pandas as pd
import os
import gc
import argparse
import pyiqa
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torchvision
import numpy as np
from datasets import load_dataset
warnings.filterwarnings("ignore")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# Image quality broken down into different characteristics, similar to CLIPIQA metric. 
# Similarity to these characteristics is measured with CLIP similarity to pairs of antonyms.
# These values can be used to train IQA-Adapter on specific characteristics of image quality rather than general IQA or IAA scores.
prompt_categories = ['brightness', 'noisiness', 'colorfulness', 'sharpness', 'contrast', 'realism']
prompt_pairs = [ 
                 'Bright image.', 'Dark image.',
                 'Clean image.', 'Noisy image.',
                 'Colorful image.', 'Dull image.',
                 'Sharp image', 'Blurry image.',
                 'High contrast image', 'Low contrast image.',
                'Photo-realistic image.', 'Cartoon style image, artificial image.'
                ]

# IQA/IAA model names from pyiqa library
#model_names = ['topiq_nr-flive', 'topiq_nr', 'topiq_nr-spaq', 'maniqa', 'clipiqa+', 'musiq-paq2piq', 'arniqa','arniqa-kadid']
model_names = ['topiq_iaa',  'dbcnn',  'laion_aes', 'liqe_mix', 'liqe', 'maniqa-pipal', 'nima-vgg16-ava', 'hyperiqa', 'tres-flive', 'arniqa-flive']
models = {
    k:pyiqa.create_metric(
            k,
            as_loss=False,
            device=device,
        ) for k in model_names
}
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
    parser = argparse.ArgumentParser(description="Script to calculate IQA/IAA values on CC3M dataset.")
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
    df = pd.DataFrame(columns=['image_file', 'text'] + model_names + prompt_categories)

    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        imgs_pil = data['images_pil']
        imgs_torch = data['images_torch']
        texts = data['img_txt']
        img_names = data['img_names']
        outputs = {}

        with torch.no_grad():
            for m in model_names:
                #print(m)
                if m == 'qalign':
                    outputs[f'{m}_quality'] = models[m](torch.clamp(imgs_torch.clone(),0,1), task_='quality')
                    outputs[f'{m}_aesthetic'] = models[m](torch.clamp(imgs_torch.clone(),0,1), task_='aesthetic')
                else:
                    outputs[m] = models[m](torch.clamp(imgs_torch.clone(),0,1))
                inputs_clip = processor(text=prompt_pairs, images=imgs_pil, return_tensors="pt", padding=True).to(device)

            outputs_clip = model(**inputs_clip).logits_per_image
            probs_clip = outputs_clip.reshape(outputs_clip.shape[0], -1, 2).softmax(dim=-1)[...,0]

        for i in range(len(img_names)):
            row = {'image_file':img_names[i], 'text':texts[i]}
            for m in outputs.keys():
                row[m] = outputs[m][i].item()
            for k, name in enumerate(prompt_categories):
                row[name] = probs_clip[i,k].item()
            df.loc[len(df)] = row
        #print(df)
    
    df.to_feather(args.ft_save_path)
    #df.to_feather(f'/test/datasets/cc3m_values/v2_train_captions_metrics_{str(range_low)}-{str(range_high)}.ft')
