# ___***IQA-Adapter***___

[![arXiv](https://img.shields.io/badge/arXiv-2412.01794-b31b1b.svg)](https://arxiv.org/abs/2412.01794)

Code for the paper "IQA-Adapter: Exploring Knowledge Transfer from Image Quality Assessment to Diffusion-based Generative Models"

*TLDR*: IQA-Adapter is a tool that combines Image Quality/Aesthetics Assessment (IQA/IAA) models with image-generation and enables quality-aware generation with diffusion-based models. It allows to condition image generators on target quality/aesthetics scores.

IQA-Adapter is based on [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) architecture.

TODO list:
- [ ] Release code for IQA-Adapter inference and training for SDXL base model
- [ ] Release weights for IQA-Adapters trained with different IQA/IAA models
- [x] Create project page
- [ ] Release code for experiments


Demonstration of guidance on quality (y-axis) and aesthetics (x-axis) scores:
![demo image](/assets/2d_viz.png)

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
