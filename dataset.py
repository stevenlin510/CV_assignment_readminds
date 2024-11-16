import os
import pathlib
import pytorchvideo.data
import imageio
from IPython.display import Image
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

class ReadMind_Dataset:
    def __init__(self, dataset_root_path="readmind_datasets", model_ckpt="MCG-NJU/videomae-base"):
        self.dataset_root_path = pathlib.Path(dataset_root_path)
        self.model_ckpt = model_ckpt

        self.image_processor = VideoMAEImageProcessor.from_pretrained(self.model_ckpt, do_center_crop=False)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            self.model_ckpt,
            label2id=self.label2id(),
            id2label=self.id2label(),
        )
        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        self.resize_to = self.get_resize_dimensions()
        self.num_frames_to_sample = self.model.config.num_frames
        self.clip_duration = self.calculate_clip_duration()

    def label2id(self):
        all_videos_paths = self.get_all_video_paths()
        class_labels = sorted({str(path).split("/")[2] for path in all_videos_paths})
        return {label: i for i, label in enumerate(class_labels)}

    def id2label(self):
        return {i: label for label, i in self.label2id().items()}

    def get_all_video_paths(self):
        return (
            list(self.dataset_root_path.glob("train/*/*.mp4")) +
            list(self.dataset_root_path.glob("val/*/*.mp4"))
        )

    def get_resize_dimensions(self):
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        return (height, width)

    def calculate_clip_duration(self):
        sample_rate = 5
        fps = 30
        return self.num_frames_to_sample * sample_rate / fps

    def get_train_transform(self):
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            Resize(self.resize_to),
                        ]
                    ),
                ),
            ]
        )

    def get_val_transform(self):
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            Resize(self.resize_to),
                        ]
                    ),
                ),
            ]
        )

def build_datasets_models(model_name):
    readmind_dataset = ReadMind_Dataset(model_ckpt=model_name)
    train_transform = readmind_dataset.get_train_transform()
    val_transform = readmind_dataset.get_val_transform()

    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(readmind_dataset.dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", readmind_dataset.clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(readmind_dataset.dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", readmind_dataset.clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    return (
        train_dataset, 
        val_dataset, 
        readmind_dataset.model, 
        readmind_dataset.image_processor
    )


def unnormalize_img(img, image_processor):
    """Un-normalizes the image pixels."""
    img = (img * image_processor.image_std) + image_processor.image_mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

def create_gif(video_tensor, image_processor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    filename = filename.split(".")[0]
    save_path = pathlib.Path("./sample_gifs/"+filename+".gif")
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy(), image_processor)
        frames.append(frame_unnormalized)
    imageio.mimsave(save_path, frames, "GIF")
    return filename


if __name__ == "__main__":
    train_dataset, val_dataset, _, image_processor = build_datasets_models("MCG-NJU/videomae-base")
    print(f"Training clips: {train_dataset.num_videos} \
            Validation clips: {val_dataset.num_videos}")
    
    sample_videos = iter(train_dataset)
    for vid in sample_videos:
        sample_video = vid
        video_tensor = sample_video["video"] 
        create_gif(video_tensor, image_processor, 
                   filename=sample_video["video_name"])
