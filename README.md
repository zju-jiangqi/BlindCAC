# Code for [Blind Computational Aberration Correction Challenge in NTIRE 2026](https://cvlai.net/ntire/2026/) </h2>

## :book: Introduction

In this challenge, we focus on blind Computational Aberration Correction (CAC), which aims to recover sharp images from observations degraded by unknown and complex lens aberrations. The problem is central to overcoming physical constraints in modern imaging systems, including mobile photography, VR and AR displays, and scientific microscopy.

This repository provides the demonstration code for training and testing in this challenge. Our open-source examples include a training case on utilizing the provided Point Spread Function (PSF) data to facilitate the training of the CAC model. Participants are encouraged to explore methods for leveraging lens-specific PSF data—available during training but unavailable during inference—to enhance model performance. Alternatively, we support training the CAC model directly using the provided paired image data without incorporating any additional PSF-based physical information. Furthermore, since constructing a comprehensive lens library is a critical research direction for the blind CAC task, except from the provided AODLibpro, participants are also permitted to use external lens databases to augment model training.

## 👊: Participation Guidelines
If you wish to participate in this challenge, please visit the official challenge homepage on [Codabench](https://www.codabench.org/competitions/12818/) for detailed rules and regulations. 

### Quick Overview: How to participate and add your model
1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1T8EF3cwMvopjudNK3Ue5bDYyqi3An4XR8mD8a3_P_yc/edit?usp=sharing) and get your team ID.
2. This project is built based on the excellent [BasicSR](https://github.com/xinntao/BasicSR) project.
   - Based on this project, you can easily modify the training pipeline in `basicsr/models/psfconstraint_model.py`, customize the network architecture in `basicsr/archs/omnilens2_arch.py`, adjust the data preprocessing logic in `basicsr/data/paired_imagepsf_dataset.py`, and provide your training configurations in `options/train/pretrain/train_PSFguided_SwinUnet_gtpsfmap.yml` as needed.
   - Once you have finalized your approach, please rename the modified files listed above as `basicsr/models/[Your_Team_ID]_[Your_Model_Name]_model.py`, `basicsr/archs/[Your_Team_ID]_[Your_Model_Name]_arch.py`, `basicsr/data/[Your_Team_ID]_[Your_Model_Name]_dataset.py`, and `options/train/pretrain/train_[Your_Team_ID]_[Your_Model_Name].yml`. This standardization is necessary to facilitate our subsequent code consolidation and review process.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
3. Put the pretrained model in `modelzoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - If the folder does not exist in the downloaded template, please create it manually.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will add your code and model checkpoint to the repository.
## <a name="setup"></a> ⚙️ Setup
The implementation of our work is based on [BasicSR](https://github.com/xinntao/BasicSR), which is an open source toolbox for image/video restoration tasks. 

- Clone this repo or download the project.
```bash
git clone https://github.com/zju-jiangqi/BlindCAC
cd BlindCAC
```

- Requirements. 
```bash
python 3.10
pytorch 1.12.1
cuda 11.3
```

```bash
conda create -n BlindCAC python=3.10
conda activate BlindCAC
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
python setup.py develop
```

## <a name="lenslib_data"></a> :book: Data Preparation
Please download the training datasets from our [Huggingface](https://huggingface.co/datasets/Strange97/OmniLens2/tree/main).
The compressed data packages required for download are categorized as follows:

1. **split_MixLib.zip, .z01, .z02, .z03:** This constitutes the primary training dataset, comprising aberration images and paired ground-truth (GT) sharp images across 3,240 distinct lenses, with 40 image pairs per lens. To extract the data, place all four volumes in the same directory and execute the unzip command on the primary .zip file; the complete dataset will be reconstructed automatically.

2. **AODLibpro_lens.zip:** This package contains physical parameters corresponding to the 3,240 lenses in the training set, including Point Spread Functions (PSF), relative illumination, and distortion map. Additionally, we provide PSFmaps—spatially aligned with the input image pixels—generated from raw PSF files via the SFR extraction method. We encourage participants to utilize these auxiliary priors to guide the model toward superior CAC performance.

### Data Instructions
#### Image Pairs
Upon completing the download of split_MixLib.zip and its corresponding multi-volume segments (.z01, .z02, and .z03), place them collectively within a specified folder.
Then, run:
```bash
unzip split_MixLib.zip
```
You will obtain a folder organized with the following structure:
```
|-- MixLib_32401l40i
  |-- hybrid
    |-- ab
      |-- 0001-40.jpg
      |-- 0001-77.jpg
      |-- 0001-303.jpg 
      ...
    |-- gt
      |-- 0001-40.jpg
      |-- 0001-77.jpg
      |-- 0001-303.jpg 
      ...    
    |-- label    
      |-- 0001-40.txt
      |-- 0001-77.txt
      |-- 0001-303.txt 
      ...     
```
``-ab``: The folder consists of images with a resolution of 1920x1280. It encapsulates the imaging results from all 3,240 lenses, with each lens applied to 40 randomly selected sharp scene images.

``-gt``: The corresponding GT images for ``-ab``, with the same resolution of 1920x1280.

``-label``: The ``.txt`` file recorded the mapping between each aberration image filename and its corresponding lens profile. The recorded lens profile can be found in the following ``AODLibpro_lens`` folder. 

Due to the large volume of training data, traversing all file paths during each training session can be time-consuming. We recommend performing the following operation to pre-generate a comprehensive list of all file paths.

Please modify the data paths in ``basicsr/data/gen_paired_csv.py``.
Then, run:
```bash
python basicsr/data/gen_paired_csv.py
```
The generated file ``meta_info.csv`` will be used in our released code.

#### Lens profile

After downloading AODLibpro_lens.zip, run:
```bash
unzip AODLibpro_lens.zip
```
You will obtain a folder organized with the following structure:
```
|-- AODLibpro_lens
  |-- Train
    |-- distort
      |-- all_rms0.01_idx9.pth
      |-- all_rms0.1_idx4999.pth
      ...
    |-- ill
      |-- all_rms0.01_idx9.pth
      |-- all_rms0.1_idx4999.pth
      ...    
    |-- psf   
      |-- all_rms0.01_idx9.pth
      |-- all_rms0.1_idx4999.pth
      ...   
    |-- psf_sfr   
      |-- all_rms0.01_idx9.pth
      |-- all_rms0.1_idx4999.pth
      ...        
```


In brief, this directory contains the physical parameters describing the aberration distributions of all 3,240 training lenses. Since geometric distortion can be corrected independently, both training and test images in this challenge are distortion-free; therefore, files in the ``-distort`` can be ignored. We encourage participants to leverage the remaining physical information to assist in model training or to re-organize the data according to your own strategies.

``-ill``:Type: Torch.tensor; Shape: [64]; Description: Relative illumination at 64 normalized fields of view from center to edge.

``-psf``:Type: Torch.tensor; Shape: [64, 3, H, W]; Description: PSFs at 64 normalized fields of view from center to edge, with 3 channels (RGB) and size H×W. The provided PSF orientation points from the image center toward the downward direction, and you need to rotate the PSFs according to the specific FoV layout to complete the simulation.

``-psf_sfr``:Type: Torch.tensor; Shape: [64, 48, 67]; Description: These are the results of rotating the raw PSFs from ``-psf`` from 0° to 360° (uniformly sampled at 48 angles). We extract Spatial Frequency Response (SFR) curves from both sagittal and tangential directions for each rotated PSF. By sampling 32 discrete points from each curve and concatenating them with the Full Width at Half Maximum (FWHM) values across the RGB channels, we construct a representative feature vector (1x1x67) for each PSF. The corresponding code in our Dataset class will automatically arrange these vectors into pixel-aligned PSF maps. For a detailed description, please refer to our reference paper, OmniLens++. This procedure is merely an example of PSF processing; we welcome participants to explore alternative methods to better utilize the raw PSF files for model guidance.

The filenames of these lens-related data files are consistent with those documented in the ``.txt`` files within the aforementioned ``-label`` directory. Participants can locate the corresponding lens profile for each image by referring its associated label.

## <a name="training"></a> :wrench: Training
### <a name="pretrain"></a> （Recommended）Train a blind CAC model with the guidance of PSF information.
The provided model directly estimates the corresponding PSF information from aberration images, supervised by ground-truth PSF maps. The predicted PSF maps are subsequently utilized to guide the CAC network. The entire pipeline is trained in an end-to-end manner. The guidelines for the training workflow are shown below.

#### Step1: Prepare Training Data
Following the instructions in [Data Preparation](#lenslib_data) to prepare training data. 

Please modify the paths to training data in `options/train/pretrain/train_PSFguided_SwinUnet_gtpsfmap.yml`:

**The lq/gt image pairs**:

`dataroot_gt: datasets/AODLibpro_img/MixLib_32401l40i/hybrid/gt`
`dataroot_lq: datasets/AODLibpro_img/MixLib_32401l40i/hybrid/ab`

**The label indicating the specific lens to which the aberration degradation of each image belongs**:

`dataroot_label: datasets/AODLibpro_img/MixLib_32401l40i/hybrid/label`

**The generated path file for the dataset:**

`csv_path: datasets/AODLibpro_img/MixLib_32401l40i/hybrid/meta_info.csv`

**The PSF maps for all the training lenses:**

`dataroot_abd: datasets/AODLibpro_lens/psf_sfr`

Our training framework will automatically retrieve paired degraded and sharp images, along with the corresponding PSF maps, for the training process.

Please also modify the validation data paths:

`dataroot_gt: datasets/MixLib_demotest/hybrid/gt`
`dataroot_lq: datasets/MixLib_demotest/hybrid/ab`

As the objective of this challenge is to achieve blind CAC, **only the aberrated images are permitted as inputs during the inference phase**.

#### Step2: Training the Blind CAC Model
run:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1 python basicsr/train.py -opt options/train/pretrain/train_PSFguided_SwinUnet_gtpsfmap.yml --auto_resume
```


### Train a simple image-to-image blind CAC model.

Taking SwinIR as an example, the Blind CAC challenge can also be implemented by directly training a model on paired lq-gt images.

#### Step1: Prepare Training Data

Following the instructions in [Data Preparation](#lenslib_data) to prepare training data. 

In this setting, only image pairs are required.

Please modify the paths to training data in `options/train/pretrain/train_SwinIR_PSNR.yml`:

#### Step2: Training the Blind CAC Model
run:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1 python basicsr/train.py -opt options/train/pretrain/train_SwinIR_PSNR.yml --auto_resume
```



## <a name="inference"></a> 💫 Inference
#### Step1: Prepare the validation/test data
Download the released data following the instructions in our challenge website.

Please modify the data paths in `options/test/test_PSFguided_SwinUnet.yml`.

#### Step2: Prepare the trained Model
Train your own blind CAC model, and save the model checkpoints.

Please modify the paths to the model checkpoints in `options/test/test_PSFguided_SwinUnet.yml`.

#### Step3: Zero-Shot Inference
run:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/test_PSFguided_SwinUnet.yml --auto_resume
```


## :smile: Develop your own model and network
This project is built based on the excellent [BasicSR](https://github.com/xinntao/BasicSR) project. 
Based on this project, you can easily modify the training pipeline in `basicsr/models/psfconstraint_model.py`, customize the network architecture in `basicsr/archs/omnilens2_arch.py`, and adjust the data preprocessing logic in `basicsr/data/paired_imagepsf_dataset.py` as needed.

## :smiley: Citation

If you find this repository useful in your project, please consider giving a :star:, and cite the corresponding paper:

```
@article{jiang2025omnilens++,
  title={OmniLens++: Blind Lens Aberration Correction via Large LensLib Pre-Training and Latent PSF Representation},
  author={Jiang, Qi and Qian, Xiaolong and Gao, Yao and Sun, Lei and Yang, Kailun and Yi, Zhonghua and Li, Wenyong and Yang, Ming-Hsuan and Van Gool, Luc and Wang, Kaiwei},
  journal={arXiv preprint arXiv:2511.17126},
  year={2025}
}

@article{jiang2024flexible,
  title={A Flexible Framework for Universal Computational Aberration Correction via Automatic Lens Library Generation and Domain Adaptation},
  author={Jiang, Qi and Gao, Yao and Gao, Shaohua and Yi, Zhonghua and Sun, Lei and Shi, Hao and Yang, Kailun and Wang, Kaiwei and Bai, Jian},
  journal={arXiv preprint arXiv:2409.05809},
  year={2024}
}
```

## :notebook: License

This project is released under the [Apache 2.0 license](LICENSE).


## :envelope: Contact

If you have any questions, please feel free to contact qijiang@zju.edu.cn.

