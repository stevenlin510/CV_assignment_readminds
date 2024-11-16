import argparse
import torch
import cv2
import numpy as np
from transformers import pipeline


def run_inference(args):
    model = pipeline("video-classification", model=args.model_name, device="cuda")

    predictions = model(args.video_path)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="LinStevenn/videomae-base-readminds", type=str, help='trained model')
    parser.add_argument("--video_path", required=True, type=str)
    args = parser.parse_args()

    predictions = run_inference(args)
    print(f"Predictions: {predictions}")
