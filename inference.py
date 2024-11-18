import cv2
import argparse
from PIL import Image
from transformers import VideoMAEImageProcessor, pipeline
from collections import Counter

def readminds_prediction(video, image_processor):
    video_cls = pipeline(model="LinStevenn/videomae-base-readminds-assignment", 
                         device="cuda",
                         image_processor=image_processor)
    results = video_cls(video)
    prediction = max(results, key=lambda x: x['score'])
    # print(f"Readminds prediction for each class: {results}")
    return prediction["label"]

def shape_detector(video):

    cap = cv2.VideoCapture(video)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frames_resized = cv2.resize(frame, (224, 224))
        frames.append(frames_resized)

    cap.release()

    test_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
                   for frame in [frames[-5], frames[-35], frames[-65]]]
    
    checkpoint = "openai/clip-vit-base-patch32"
    detector = pipeline(model=checkpoint, 
                        task="zero-shot-image-classification",
                        device="cuda")
    
    results = detector(test_images, candidate_labels=["triangle", "star", "wave", "square"])
    
    # print(f"Shape prediction based on last 2 seconds: {results}")
    
    frame1_prediction = max(results[0], key=lambda x: x['score'])['label']
    frame2_prediction = max(results[1], key=lambda x: x['score'])['label']
    frame3_prediction = max(results[2], key=lambda x: x['score'])['label']

    # Voting mechanism
    predictions = [frame1_prediction, frame2_prediction, frame3_prediction]
    final_prediction = Counter(predictions).most_common(1)[0][0]
    return final_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="LinStevenn/videomae-base-readminds-assignment", type=str, help='trained model')
    parser.add_argument("--video_path", required=True, type=str)
    args = parser.parse_args()
    
    image_processor = VideoMAEImageProcessor.from_pretrained(
        "LinStevenn/videomae-base-readminds-assignment", 
 #       local_files_only=True)
        )
    
    image_processor.size = {"height": 224, "width": 224}

    prediction = readminds_prediction(video=args.video_path, 
                                      image_processor=image_processor)
    print("="*(30+len(prediction)))
    print(f"| Final readmind prediction: {prediction}|")

    shape_detection = shape_detector(args.video_path)
    print("="*(27+len(shape_detection)))
    print(f"| Final shape prediction: {shape_detection}|")
    print("="*(27+len(shape_detection)))

