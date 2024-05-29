# CTIDiffusion

This repository contains the implementation of the paper titled "Conditional Text Image Generation with Diffusion Models" (arXiv:2306.10804v1). This project focuses on generating text images using diffusion models under various conditions, including image condition, text condition, and style condition, to control the attributes, contents, and styles of the generated samples.

## Overview

In this work, the authors proposed a novel approach for text image generation by leveraging the power of diffusion models. The method, Conditional Text Image Generation with Diffusion Models (CTIG-DM), allows for the creation of high-quality and diverse text images under different conditional settings. The conditions explored in this project include:

- **Image Condition**: Using existing images as a basis for generating new text images.
- **Text Condition**: Generating images based on given text descriptions.
- **Style Condition**: Controlling the style attributes of the generated text images.

The paper explore four distinct modes for text image generation:

1. **Synthesis Mode**: Creating entirely new text images from scratch.
2. **Augmentation Mode**: Enhancing existing images with additional text or modifying the text within the image.
3. **Recovery Mode**: Reconstructing missing or corrupted parts of text images.
4. **Imitation Mode**: Mimicking the style and content of a given reference text image.

## Dataset Used

For training and evaluation, As mentioned in the paper I utilized the IAM Handwritten Dataset, which is a widely used dataset in the field of handwritten text recognition and generation. The IAM Handwritten Dataset contains a large collection of handwritten text images along with their corresponding transcriptions. 

### Dataset Details

- **Source**: The IAM Handwritten Dataset is publicly available on Kaggle at "https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database"
- **Content**: This dataset comprises images of handwritten words along with their ground-truth transcriptions. It provides a rich resource for training models on handwriting recognition and generation tasks.
- **Usage in Paper**: The dataset is employed to train and evaluate the diffusion models presented in the paper, ensuring the generated text 
images are both realistic and diverse.

<!-- Commented code -->
<!-- 
## Getting Started

To get started with the CTIDiffusion implementation, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/your-repo/CTIDiffusion.git
    cd CTIDiffusion
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the Dataset**:
    - Access the IAM Handwritten Dataset on Kaggle and download it.
    - Extract the dataset and place it in the appropriate directory as specified in the code.

4. **Train the Model**:
    ```sh
    python train.py --config configs/train_config.yaml
    ```

5. **Generate Text Images**:
    ```sh
    python generate.py --config configs/generate_config.yaml
    ```

## Results

Our experiments demonstrate that the proposed CTIG-DM method can effectively generate high-quality and diverse text images under different conditional settings. The generated samples are evaluated based on various metrics, including image quality, diversity, and the ability to control specific attributes through conditioning.

## Citation

If you find our work useful in your research, please consider citing the paper:

```bibtex
@article{ctidiffusion2023,
  title={Conditional Text Image Generation with Diffusion Models},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:2306.10804v1},
  year={2023}
}
-->