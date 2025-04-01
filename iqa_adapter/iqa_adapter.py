import os
from typing import List
import torch.nn.functional as F
import torch
from diffusers.pipelines.controlnet import MultiControlNetModel
from safetensors import safe_open

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor

# code is based on existing adapters for other tasks: 
# https://github.com/tencent-ailab/IP-Adapter
# https://github.com/TencentARC/T2I-Adapter



class ProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, input_embed_dims=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(input_embed_dims, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, input_dim=1, inner_dim=128, clip_extra_context_tokens=2):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens_num = clip_extra_context_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, inner_dim),
            torch.nn.GELU(),
            torch.nn.Linear(inner_dim, cross_attention_dim * clip_extra_context_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens_num, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IQAAdapter:
    # class for SD-1.5 architecture
    # used as a parent for SDXL class
    def __init__(self, sd_pipe, ip_ckpt, 
                device, num_tokens=4, input_dim=3, proj_type='linear',
                mlp_inner_dim=128, load_pretrained=True, adapter_dtype=torch.float16,
                enable_neg_guidance=False, neg_guidance_scale=0.5):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.pipe = sd_pipe.to(self.device)
        self.adapter_dtype = adapter_dtype
        self.enable_neg_guidance = enable_neg_guidance
        self.neg_guidance_scale = neg_guidance_scale
        self.set_iqa_adapter(adapter_dtype)

        self.proj_type = proj_type
        self.image_proj_model = self.init_proj(proj_type, mlp_inner_dim, adapter_dtype)
        if load_pretrained:
            self.load_iqa_adapter()
        else:
            print('Skipped loading weights, ip_ckpt ignored')
        #self.load_ip_adapter()

    def init_proj(self, proj_type, mlp_inner_dim, adapter_dtype):
        if proj_type == 'linear':
            proj_model = ProjModel(
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                input_embed_dims=self.input_dim,
                clip_extra_context_tokens=self.num_tokens,
            ).to(self.device, dtype=adapter_dtype)
        elif proj_type == 'mlp':
            proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            input_dim=self.input_dim,
            inner_dim=mlp_inner_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=adapter_dtype)
        return proj_model

    def set_iqa_adapter(self, adapter_dtype):
        unet = self.pipe.unet
        attn_procs = {}
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
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=adapter_dtype)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_iqa_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        print('loaded')
    
    @torch.inference_mode()
    def get_iqa_embeds(self, target_quality_vals=None, enable_neg_guidance=False, neg_guidance_scale=0.5): # [bs, 3]
        target_quality_vals = target_quality_vals.to(self.device, dtype=self.adapter_dtype)
        image_prompt_embeds = self.image_proj_model(target_quality_vals)
        if enable_neg_guidance:
            neg_embeds = target_quality_vals.to(self.device, dtype=self.adapter_dtype).clone()
            neg_embeds *= -neg_guidance_scale
            uncond_image_prompt_embeds = self.image_proj_model(neg_embeds)
        else:
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(target_quality_vals))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        target_quality_vals=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        prompt_embeds=None,
        **kwargs,
    ):
        self.set_scale(scale)
        num_prompts = len(target_quality_vals)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        quality_embeds, uncond_quality_embeds = self.get_iqa_embeds(
            target_quality_vals=target_quality_vals,
            enable_neg_guidance=self.enable_neg_guidance,
            neg_guidance_scale=self.neg_guidance_scale
        ) # [bs, num tokens, 768]
        bs_embed, seq_len, _ = quality_embeds.shape
        quality_embeds = quality_embeds.repeat(1, num_samples, 1)
        quality_embeds = quality_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_quality_embeds = uncond_quality_embeds.repeat(1, num_samples, 1)
        uncond_quality_embeds = uncond_quality_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            if prompt_embeds is not None:
                prompt_embeds_ = prompt_embeds
            
            prompt_embeds_local = torch.cat([prompt_embeds_, quality_embeds], dim=1)
            negative_prompt_embeds_local = torch.cat([negative_prompt_embeds_, uncond_quality_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds_local,
            negative_prompt_embeds=negative_prompt_embeds_local,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IQAAdapterXL(IQAAdapter):
    """IQA-Adapter for SDXL"""

    def generate(
        self,
        target_quality_vals,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = len(target_quality_vals)

        if prompt is None:
            prompt = "best quality, high quality"
        #if negative_prompt is None:
        #    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List) and not (negative_prompt is None):
            negative_prompt = [negative_prompt] * num_prompts

        quality_embeds, uncond_quality_embeds = self.get_iqa_embeds(target_quality_vals, enable_neg_guidance=self.enable_neg_guidance,
                                                                              neg_guidance_scale=self.neg_guidance_scale)
        bs_embed, seq_len, _ = quality_embeds.shape
        quality_embeds = quality_embeds.repeat(1, num_samples, 1)
        quality_embeds = quality_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_quality_embeds = uncond_quality_embeds.repeat(1, num_samples, 1)
        uncond_quality_embeds = uncond_quality_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds_,
                negative_prompt_embeds,
                pooled_prompt_embeds_,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            if prompt_embeds is None:
                prompt_embeds = prompt_embeds_
            if pooled_prompt_embeds is None:
                pooled_prompt_embeds = pooled_prompt_embeds_
            
            prompt_embeds = torch.cat([prompt_embeds, quality_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_quality_embeds], dim=1)

        #self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            #generator=self.generator,
            **kwargs,
        ).images

        return images





class RefBasedIQAAdapter:
    # class for SD-1.5 architecture
    # used as a parent for SDXL class
    def __init__(self, sd_pipe, ip_ckpt, 
                device, num_tokens=4, input_dim=3, proj_type='linear',
                mlp_inner_dim=128, load_pretrained=True, adapter_dtype=torch.float16, enable_neg_guidance=False):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.pipe = sd_pipe.to(self.device)
        self.enable_neg_guidance = enable_neg_guidance
        self.adapter_dtype = adapter_dtype
        self.set_iqa_adapter(adapter_dtype)
        self.proj_type = proj_type
        self.image_proj_model = self.init_proj(proj_type, mlp_inner_dim, adapter_dtype)
        if load_pretrained:
            self.load_iqa_adapter()
        else:
            print('Skipped loading weights, ip_ckpt ignored')
            
    def init_proj(self, proj_type, mlp_inner_dim, adapter_dtype):
        if proj_type == 'linear':
            proj_model = ProjModel(
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                input_embed_dims=self.input_dim,
                clip_extra_context_tokens=self.num_tokens,
            ).to(self.device, dtype=adapter_dtype)
        elif proj_type == 'mlp':
            proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            input_dim=self.input_dim,
            inner_dim=mlp_inner_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=adapter_dtype)
        return proj_model

    def set_iqa_adapter(self, adapter_dtype):
        unet = self.pipe.unet
        attn_procs = {}
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
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=adapter_dtype)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_iqa_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        print('loaded')
    
    @torch.inference_mode()
    def get_iqa_embeds(self, target_img_embeds=None, neg_img_embeds=None, enable_neg_guidance=False): 
        target_img_embeds = target_img_embeds.to(self.device, dtype=self.adapter_dtype)
        image_prompt_embeds = self.image_proj_model(target_img_embeds)
        if enable_neg_guidance:
            neg_embeds = target_img_embeds.to(self.device, dtype=self.adapter_dtype).clone()
            uncond_image_prompt_embeds = self.image_proj_model(neg_embeds)
        else:
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(target_img_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        target_img_embeds=None,
        neg_img_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        prompt_embeds=None,
        **kwargs,
    ):
        self.set_scale(scale)
        num_prompts = len(target_img_embeds)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        quality_embeds, uncond_quality_embeds = self.get_iqa_embeds(
            target_img_embeds=target_img_embeds,
            neg_img_embeds=neg_img_embeds,
            enable_neg_guidance=self.enable_neg_guidance,
        ) # [bs, num tokens, 768]
        bs_embed, seq_len, _ = quality_embeds.shape
        quality_embeds = quality_embeds.repeat(1, num_samples, 1)
        quality_embeds = quality_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_quality_embeds = uncond_quality_embeds.repeat(1, num_samples, 1)
        uncond_quality_embeds = uncond_quality_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            if prompt_embeds is not None:
                prompt_embeds_ = prompt_embeds
            
            prompt_embeds_local = torch.cat([prompt_embeds_, quality_embeds], dim=1)
            negative_prompt_embeds_local = torch.cat([negative_prompt_embeds_, uncond_quality_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds_local,
            negative_prompt_embeds=negative_prompt_embeds_local,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
    


class RefBasedIQAAdapterXL(RefBasedIQAAdapter):
    """IQA-Adapter for SDXL"""

    def generate(
        self,
        target_quality_vals,
        neg_img_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = len(target_quality_vals)

        if prompt is None:
            prompt = "best quality, high quality"
        #if negative_prompt is None:
        #    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List) and not (negative_prompt is None):
            negative_prompt = [negative_prompt] * num_prompts

        quality_embeds, uncond_quality_embeds = self.get_iqa_embeds(target_quality_vals, enable_neg_guidance=self.enable_neg_guidance,
                                                                              neg_img_embeds=neg_img_embeds)
        bs_embed, seq_len, _ = quality_embeds.shape
        quality_embeds = quality_embeds.repeat(1, num_samples, 1)
        quality_embeds = quality_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_quality_embeds = uncond_quality_embeds.repeat(1, num_samples, 1)
        uncond_quality_embeds = uncond_quality_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds_,
                negative_prompt_embeds,
                pooled_prompt_embeds_,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            if prompt_embeds is None:
                prompt_embeds = prompt_embeds_
            if pooled_prompt_embeds is None:
                pooled_prompt_embeds = pooled_prompt_embeds_
            
            prompt_embeds = torch.cat([prompt_embeds, quality_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_quality_embeds], dim=1)

        #self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            #generator=self.generator,
            **kwargs,
        ).images

        return images