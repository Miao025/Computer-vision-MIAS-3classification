# Computer Vision MIAS 3-class classification 

This project develops a 3-class classification system for breast cancer using the Mammographic Image Analysis Society (MIAS) dataset. The goal is to classify mammographic images into three categories: Normal, Benign, and Malignant, with a focus on the accuracy and recall for each class.

The project addresses challenges such as the small dataset size and class imbalance through data augmentation, upsampling, and adjusteing class weights. To improve image quality and focus on relevant features, multiple image preprocessing steps are taken and a fusion of hybrid deep features (FHDF) from fine-tuned VGG16, VGG19, ResNet50, and DenseNet121 is used, inspired by [Chakravarthy et al. (2024)](https://link.springer.com/article/10.1007/s44196-024-00593-7) and [Hsieh et al. (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0952197612002813).

## Project Structure

```
main.ipynb
Dataset/
    mias_derived_info.csv
    MIAS/
        mdb001.png
        mdb002.png
        ...
utils/
    dataset.py
    evaluate.py
    models.py
    preprocess.py
    train.py
```

## Dataset
The MIAS dataset consists of 322 greyscale digital mammographic images, categorized as follows:
- Normal: 207 images
- Benign: 64 images
- Malignant: 51 images

Label information is stored in csv.

## Scripts
- Image preprocessing functions: [`utils/preprocess.py`](utils/preprocess.py)
- Dataset setup for Pytorch: [`utils/dataset.py`](utils/dataset.py)
- Model configurations: [`utils/models.py`](utils/models.py)
- Training and evaluation: [`utils/train.py`](utils/train.py), [`utils/evaluate.py`](utils/evaluate.py)

## Features
- Data preprocessing
  - Flip images for consistency
  - Crop black blocks on both sides
  - Salt-and-pepper noise removal through adaptive median filter
  - Contrast enhancement through Contrast Limited Adaptive Histogram Equalization (CLAHE)
  - Edge detection through Canny Edge Detection with Sobel Filter
  - Pectoral muscle line detection based on edges, through Hough Transform with customized standards
  - Pectoral muscle region removal
  - Offline augmentation by filp each image vertically and horizontally and rotate each image at 45°, 90°, 135°, 180°, 235°, and 270°.
  
- Model
  - Each pretrained model (VGG16, VGG19, ResNet50, DenseNet121) fine tuned partialy on MIAS dataset.
  - Features from the penultimate layer of each model extracted and concatenated.
  - Concatenated features processed through a fully connected network (Early fusion technique).

- Train
  - 5-fold train-val to asses the general model performance.
  - Adam optimizer, learning rate scheduler and mini-batch techniques to improve training.
  - Cross-Entropy Loss with adjusted class weights and upsampling during training to reduce class imbalance.

## Setup
**1. Clone and navigate to the repo:**
```cli
git clone https://github.com/Miao025/Computer-vision-MIAS-3classification

cd Computer-vision-MIAS-3classification
```

**2. Create and activate a virtual environment:**
```
conda create --name <your-env>
conda activate <your-env>
```

**3. Install dependencies:**
```
pip install -r requirements.txt
```
*Note that a NVIDIA GPU  is required for this project. Check the CUDA version and change the 'cupy-cuda12x' in requirements.txt to the accordingly version.*

## Usage

Run `main.ipynb` to and explore step-by-step image preprocessing and the model's training process.

## Contributor
Miao

## Citation

If you use this code, please cite the MIAS dataset and the referenced papers [Chakravarthy et al. (2024)](https://link.springer.com/article/10.1007/s44196-024-00593-7) and [Hsieh et al. (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0952197612002813)