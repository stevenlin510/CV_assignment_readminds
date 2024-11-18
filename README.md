# Assignment 1: I Can Read Your Mind

## Description
This assignment is part of the Computer Vision course for the Fall 2024 semester. The ultimate goal of the program is to determine the shape drawn by the user.

## Installation

Ensure that the `python` environment is installed. Install the required packages using the following command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip3 install -r requirements.txt
```

## Training
To fine-tune the pre-trained video classification model, run the following command in the terminal:

```bash
python main.py --model_name <model-name>
```
- You can replace `<model-name>` with any desired pre-trained model (e.g., `MCG-NJU/videomae-base`).
- Please ensure that the HuggingFace access token is initialized from your local computer. 

```bash
huggingface-cli login
```

- For shape detection, zero-shot image classification is used, so there is no need for additional training. 

## Inference
During inference, the readminds prediction and shape detection will be conducted. Streaming video is not yet supported. 

```bash
python inference.py --video <video_path>
```

### Dataset Structure
The dataset, which contains videos from Prof. Baltes and our own videos, is organized as follows:
```
readmind_datasets/
    train/
        triangle/
            video1.mp4
            video2.mp4
            ...
        star/
            video1.mp4
            video2.mp4
            ...
        square/
            video1.mp4
            video2.mp4
            ...
        wave/
            video1.mp4
            video2.mp4
            ...
    val/
        triangle/
            video1.mp4
            ...
        star/
            video1.mp4
            ...
        ...
```
- Please download the dataset from the sharable link.
- Sample videos are available [here](https://github.com/stevenlin510/CV_assignment_readminds/tree/main/sample_gifs).

## Acknowledgements
This assignment is based on the codebase from [video_classification](https://huggingface.co/docs/transformers/tasks/video_classification) and [image_classifcation](https://huggingface.co/docs/transformers/tasks/zero_shot_image_classification)
