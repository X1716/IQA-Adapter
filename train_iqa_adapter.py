import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from datasets import load_dataset
import io
# based on https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train_sdxl.py
from iqa_adapter.iqa_adapter import ProjModel, MLPProjModel
from iqa_adapter.iqa_adapter import is_torch2_available
if is_torch2_available():
    from iqa_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from iqa_adapter.attention_processor import IPAttnProcessor, AttnProcessor


def positional_enc(p, L=10, input_dim=2):
        trig_args = (torch.pi * 2.**torch.linspace(0, L - 1, L, device=p.device)).unsqueeze(0).repeat((input_dim, 1))
        trig_args = p[:,:, None] * trig_args[None, :, :]
        sin_tensor = torch.sin(trig_args).reshape([p.shape[0], -1])
        cos_tensor = torch.cos(trig_args).reshape([p.shape[0], -1])
        res = torch.concat([p, sin_tensor, cos_tensor], dim=-1)
        return res.reshape((-1, L * p.shape[1] * 2 + input_dim))
# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.01, ti_drop_rate=0.01, image_root_path="",
                 use_flag=False, qual_columns=['musiq-koniq', 'musiq-ava'], normalized=True, train_dset_name='CC3M', hf_cache_dir='./', quality_data_path=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.use_flag = use_flag
        self.qual_columns = qual_columns
        self.normalized = normalized
        self.train_dset_name = train_dset_name
        if self.train_dset_name == 'CC3M':
            # put your relevant HF cache directory here. CC3M takes about 1.1TB of space
            self.data = load_dataset("pixparse/cc3m-wds", cache_dir=hf_cache_dir, split='train')
        elif self.train_dset_name == 'LAION-SBS':
            self.data = load_dataset("bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images", cache_dir=hf_cache_dir)

        self.quality_data = pd.read_feather(quality_data_path)
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __getitem__(self, idx):
        item = self.data['train'][idx] 
        text = item["text"]
        if self.train_dset_name == 'LAION-SBS':
            image_name = str(idx)
        else:
            image_name = item['__key__']

        quality_item = self.quality_data.loc[self.quality_data.image_file == image_name]
        # prepare quality vector
        if self.normalized:
            quality_vec = [float(quality_item[f'{x}_normalized'].values[0]) for x in self.qual_columns]
        else:
            quality_vec = [float(quality_item[x].values[0]) for x in self.qual_columns]
        
        if self.use_flag:
            quality_vec.append(1.0)
        quality_vec = torch.tensor(quality_vec)


        # read image
        if self.train_dset_name == 'LAION-SBS':
            raw_image = Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB')
        else:
            raw_image = item['jpg'].convert('RGB')
        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        
        # drop
        drop_qual_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_qual_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_qual_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "drop_qual_embed": drop_qual_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            "quality_vec":quality_vec
        }
        
    
    def __len__(self):
        return len(self.data['train'])
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    drop_qual_embeds = [example["drop_qual_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    quality_vecs = torch.stack([example["quality_vec"] for example in data])
    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "drop_qual_embeds": drop_qual_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "quality_vecs": quality_vecs
    }
    

class IQAAdapter(torch.nn.Module):
    """IQA-Adapter"""
    def __init__(self, unet, qual_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.qual_proj_model = qual_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            print(f'Loading pretrained IQA adapter from checkpoint: {ckpt_path}')
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        iqa_tokens = self.qual_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, iqa_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_q_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.qual_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.qual_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_q_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.qual_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_q_proj_sum != new_q_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for IQA-Adapter (non-reference-based).")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained IQA-Adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=False,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=False,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--use_flag", type=int, default=1, help="==1 to add 1.0 constant feature to differentiate from unconditional generation.")
    parser.add_argument("--normalized", type=int, default=1, help="==1 to use normalized features")
    parser.add_argument("--num_tokens", type=int, default=2, help="number of tokens to project quality embedding to.")
    parser.add_argument("--features", nargs='*', type=str, default=['brightness', 'noisiness', 'colorfulness', 'sharpness', 'contrast'], help="Quality features to use.")
    parser.add_argument("--pos_encode", type=int, default=0, help="==1 to add sine/cosine positional encoding")
    parser.add_argument("--pos_encode_L", type=int, default=10, help="number of sines/cosines in positional encoding.")
    parser.add_argument("--proj_type", type=str, default='linear', choices=['linear', 'mlp'], help="number of sines/cosines in positional encoding.")
    parser.add_argument("--mlp_inner_dim", type=int, default=128, help="MLP inner layer dim.")
    parser.add_argument("--train_dataset", type=str, default='CC3M', help="Name of the dataset for training", choices=['CC3M', 'LAION-SBS'])
    parser.add_argument("--quality_data_path", type=str, default=None, help="Path to a feather file with metric values for the training set of images")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    train_dset_name = args.train_dataset

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    #IQA-Adapter
    num_tokens = args.num_tokens
    quality_features = args.features
    if len(args.features) == 1 and ' ' in args.features[0]:
        quality_features = args.features[0].split(' ')
    use_flag = args.use_flag == 1
    normalized = args.normalized == 1
    input_embed_dim = len(quality_features)
    if use_flag:
        input_embed_dim += 1
    proj_type = args.proj_type
    print(f'========= NUM TOKENS: {num_tokens} =========')
    print(f'========= FEATURES: {quality_features} =========')
    print(f'========= USE FLAG: {use_flag} =========')
    print(f'========= NORMALIZED: {normalized} =========')
    print(f'========= EMBEDDING DIM: {input_embed_dim} =========')
    print(f'========= PROJ TYPE: {proj_type} =========')
    print(f'========= LR: {args.learning_rate} =========')
    pos_encode_needed = False
    original_input_dim = input_embed_dim
    if args.pos_encode:
        pos_encode_needed = True
        L = args.pos_encode_L
        input_embed_dim = input_embed_dim * 2 * L + input_embed_dim
        print(f'========= EMBEDDING DIM AFTER POS ENCODE: {input_embed_dim}, L={L} =========')
    
    if proj_type == 'linear':
        proj_model = ProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=input_embed_dim,
            clip_extra_context_tokens=num_tokens,
        )
    elif proj_type == 'mlp':
        proj_model = MLPProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            input_dim=input_embed_dim,
            inner_dim=args.mlp_inner_dim,
            clip_extra_context_tokens=num_tokens,
        )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    iqa_adapter = IQAAdapter(unet, proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    
    # optimizer
    params_to_opt = itertools.chain(iqa_adapter.qual_proj_model.parameters(),  iqa_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path,
                               qual_columns=quality_features, use_flag=use_flag, normalized=normalized, 
                               train_dset_name=train_dset_name, quality_data_path=quality_data_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    iqa_adapter, optimizer, train_dataloader = accelerator.prepare(iqa_adapter, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(iqa_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    image_embeds = batch["quality_vecs"]
                    if pos_encode_needed:
                        image_embeds = positional_enc(image_embeds, L=L, input_dim=original_input_dim)
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_qual_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
            
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                        
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = iqa_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and step % 25 == 0:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
