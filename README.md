# Image Colorization

This is a web app that use CNN to recover colored image from black and white.

## Prerequisites

```
pip install -r requirements
```

The requirements are for production. To work on a gpu, please install version of torch and torchvision

## Model Architecture

The model used for evaluation is a Convolution Neural Network and the full version is created by [this repo](https://github.com/lukemelas/Automatic-Image-Colorization/) . There are two relative new idea added beyond basic CNN architecture:

(1) It used pretrained model Resnet-18 as backbone
(2) During the data pre-processing phase, the training data is converted from RGB to LAB colorspace (Lightness, A and B). By this mean the separation process is easier and faster since seperating Lightness channel out as the input greyscale images is more convenient.

## Dataset

The dataset used for training and validating is from [wandb competition](https://wandb.ai/wandb/colorizer-applied-dl/benchmark). This is not a huge dataset compared with the original dataset used for this model. Due to time and resources limit, the model deployed has to be small enough. The dataset contains mostly flowers and other natural views.

## Deployment

This app's templates are written in HTML, CSS and Javascript. Additionally, some bootstrap features are also added. It is deployed to heroku and available at fsdl-colorization.heroku.com

## Reference

https://lukemelas.github.io/image-colorization.html

https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html

