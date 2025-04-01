# ___***IQA-Adapter***___

[![arXiv](https://img.shields.io/badge/arXiv-2412.01794-b31b1b.svg)](https://arxiv.org/abs/2412.01794)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](https://x1716.github.io/IQA-Adapter/)


Code for the paper "IQA-Adapter: Exploring Knowledge Transfer from Image Quality Assessment to Diffusion-based Generative Models"

*TLDR*: IQA-Adapter is a tool that combines Image Quality/Aesthetics Assessment (IQA/IAA) models with image-generation and enables quality-aware generation with diffusion-based models. It allows to condition image generators on target quality/aesthetics scores.

IQA-Adapter builds upon [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) architecture.

TODO list:
- [x] Release code for IQA-Adapter inference and training for SDXL base model (in progress)
- [x] Release weights for IQA-Adapters trained with different IQA/IAA models (in progress)
- [x] Create project page
- [ ] Release code for experiments


Demonstration of guidance on quality (y-axis) and aesthetics (x-axis) scores:
![demo image](/assets/2d_viz.png)


## Run IQA-Adapter

### Prerequisites

First, clone this repository:

      git clone https://github.com/X1716/IQA-Adapter.git

Next, create a virtual environment, e.g. with anaconda:

      conda create --name iqa_adapter python=3.12.2
      conda activate iqa_adapter

Install PyTorch suitable for your CUDA/ROCm/MPS device:

      pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 
      # for CUDA 12.1

Newer Python and PyTorch versions should also work.

Install other requirements for this project:

      pip install -r requirements.txt

### Demo 

To test a pretrained IQA-Adapter you can check out [demo_adapter.ipynb](./demo_adapter.ipynb) jupyter notebook. The weights for the IQA-Adapter can be downloaded from [here](https://drive.google.com/drive/folders/1jVYM96nbk0pUV4HSHiUzWGlTSLg-dv5h?usp=sharing) (Google Drive).

### Training script

train_iqa_adapter.py can be used to train/fine-tune IQA-Adapter. We trained it on a SLURM cluster with slurm_train_script.sh. Train job can be dispatched with:

      sbatch slurm_train_script.sh
Note that this script should be modified for your particular cluster setup (e.g., paths to input/output directories, pyXis container and other things should be specified). It is configured for distributed training with 5 nodes and 8 GPUs per node.

## Citation
If you find this work useful for your research, please cite us as follows:
```bibtex
@misc{iqaadapter,
      title={IQA-Adapter: Exploring Knowledge Transfer from Image Quality Assessment to Diffusion-based Generative Models}, 
      author={Khaled Abud and Sergey Lavrushkin and Alexey Kirillov and Dmitriy Vatolin},
      year={2024},
      eprint={2412.01794},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.01794}, 
}
```
