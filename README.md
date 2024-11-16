# Assignment 1: I can Read Your Mind

## Description
This project implements a video classification model using the VideoMAE architecture. It mainly leverages the Hugging Face Transformers library to train a model on a custom dataset of videos. The task is designed to classify videos into different categories based on their drawing.

## Installation Instructions

Ensure `python` environment is installed. Install the required packages:

```bash
pip3 install -r requirements.txt
```

## Training
To finetune the pre-trained video classification model, run the following command in the terminal:

```bash
python main.py --model_name <model-name>
```
- Replace `<model-name>` with the pretrained model (e.g., `MCG-NJU/videomae-base`).

## Inference
During the inference, we take a video input. The streaming video is not yet supported. 

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

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
