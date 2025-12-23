# Learning Differential Pyramid Representation for Tone Mapping (DPRNet) — NeurIPS 2025

This repository provides the official implementation of **DPRNet (Differential Pyramid Representation Network)** for high-fidelity HDR-to-LDR tone mapping.

## **[Online Demo](https://xxxxxxdprnet.github.io/DPRNet/)** 



## 🔥 Overview

Existing tone mapping methods often operate on downsampled inputs and rely on handcrafted pyramids to recover high-frequency details. These designs typically fail to preserve fine textures and structural fidelity in complex HDR scenes. Furthermore, most methods lack an effective mechanism to jointly model **global tone consistency** and **local contrast enhancement**, which can lead to globally flat results or locally inconsistent outputs (e.g., halo artifacts).

We propose **DPRNet**, an end-to-end framework that introduces a **learnable differential pyramid** generalizing Laplacian and Difference-of-Gaussian (DoG) pyramids through **content-aware differencing across scales**, enabling adaptive capture of high-frequency variations under diverse luminance/contrast conditions. DPRNet further integrates **global tone perception** and **local tone tuning** modules operating on downsampled inputs for efficient yet expressive tone adaptation. Finally, an **iterative detail enhancement** module progressively restores full-resolution outputs in a coarse-to-fine manner, reinforcing structure and sharpness.




## 📦 Dataset: HDRI Haven 

The paper introduces a new dataset **HDRI Haven**, available via Baidu Netdisk:

- **Baidu Netdisk:** [Download HDRI Haven](https://pan.baidu.com/s/1i4n9OOvzud84vfc4wVBtXQ)
- **Extraction code:** `srly`



## 🧩 Pretrained Weights

Pretrained training weights can be downloaded via Baidu Netdisk:

- **Baidu Netdisk:** [Download pretrained weights](https://pan.baidu.com/s/1ZSM891Hu15ujV9HOnPnnDA)
- **Extraction code:** `64ou`




## 🏋️ Training

Run training with the provided config:

```bash
python train.py -opt options/train/train_DPRNet_hdrplus4k.yml
```



## 🧪 Testing

Run testing with the provided config:

```bash
python test_LP3DLUT.py -opt options/test/test_DPRNet_hdrplus4k.yml
```

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{yang2024learning,
  title={Learning differential pyramid representation for tone mapping},
  author={Yang, Qirui and Li, Yinbo and Liu, Yihao and Jiang, Peng-Tao and Zhang, Fangpu and Cheng, Qihua and Yue, Huanjing and Yang, Jingyu},
  journal={arXiv preprint arXiv:2412.01463},
  year={2024}
}



