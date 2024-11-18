# Assignment 1: I can Read Your Mind

## Description
This is an assignment from Computer Vision course in 2024 Fall semester. The ultimate goal for the program is able to determine the shape that is drawn by the user.

## Installation Instructions

Ensure `python` environment is installed. Install the required packages:

```bash
pip3 install -r requirements.txt
```

## Train
To finetune the pre-trained video classification model, run the following command in the terminal:

```bash
python main.py --model_name <model-name>
```
- You can replace `<model-name>` with any desired pretrained model (e.g., `MCG-NJU/videomae-base`).
- Please ensure the HuggingFace access token is initiate from your local computer. 
```bash
huggingface-cli login
```
- For shape detection, zero-shot image classification is used, therefore there is no need to train. 

## Inference
During the inference, the readminds prediction and the shape detection will be conducted. The streaming video is not yet supported. 

```bash
python inference.py --video <video_path>
```

### Dataset Structure
The dataset that contains Prof. Baltes's videos and ours videos is organized as follows:
```
readmind_datasets/
    train/
        triangle/
            video1.mp4
            video2.mp4
        star/
            video1.mp4
            video2.mp4
        square/
            video1.mp4
            video2.mp4
        wave/
            video1.mp4
            video2.mp4
    val/
        triangle/
            video1.mp4
        star/
            video1.mp4
        ...
```
- Sample videos are given [here](https://github.com/stevenlin510/CV_assignment_readminds/tree/main/sample_gifs).

## Acknowledgement
This assignment is based on the codebase from [video_classification](https://huggingface.co/docs/transformers/tasks/video_classification) and [image_classifcation](https://huggingface.co/docs/transformers/tasks/zero_shot_image_classification)
