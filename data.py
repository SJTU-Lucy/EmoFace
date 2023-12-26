import os
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import Wav2Vec2Processor
import torch
import librosa
import numpy as np

fps = 60
label_dim = 174
emotion_id = {"ang": 0, "dis": 1, "fea": 2, "hap": 3, "neu": 4, "sad": 5, "sur": 6}


class AudioDataProcessor:
    def __init__(self, sampling_rate=16000) -> None:
        self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self._sampling_rate = sampling_rate

    def run(self, audio):
        speech_array, sampling_rate = librosa.load(audio, sr=self._sampling_rate)
        input_values = np.squeeze(self._processor(speech_array, sampling_rate=sampling_rate).input_values)
        return input_values

    @property
    def sampling_rate(self):
        return self._sampling_rate


class FeaturesConstructor:
    def __init__(self, audio_max_duration=60):
        self._audio_max_duration = audio_max_duration
        self._audio_data_processor = AudioDataProcessor()
        self._audio_sampling_rate = self._audio_data_processor.sampling_rate

    def infer_run(self, audio):
        audio_data = self._audio_data_processor.run(audio)
        feature_indices = list(range(0, len(audio_data), self._audio_sampling_rate * self._audio_max_duration))[1:]
        audio_chunks = np.split(audio_data, feature_indices)
        feature_chunks = []
        for chunk in audio_chunks:
            seq_len = int(len(chunk) / self._audio_sampling_rate * fps)
            label_chunk = np.zeros((seq_len, label_dim))
            feature_chunks.append([chunk, label_chunk])
        return feature_chunks

    def train_run(self, audio_path, label_path):
        audio_data = self._audio_data_processor.run(audio_path)
        feature_indices = list(range(0, len(audio_data), self._audio_sampling_rate * self._audio_max_duration))[1:]
        audio_chunks = np.split(audio_data, feature_indices)

        label_data = np.loadtxt(label_path, delimiter=",")
        total_duration = audio_data.size / self._audio_sampling_rate
        label_data = label_data[: int(fps * total_duration)]

        label_indices = list(range(0, len(label_data), fps * self._audio_max_duration))[1:]
        label_chunks = np.split(label_data, label_indices)

        feature_chunks = []
        for audio_chunk, label_chunk in zip(audio_chunks, label_chunks):
            feature_chunks.append([audio_chunk, label_chunk])
        return feature_chunks, label_chunks


class FeatureDataset(Dataset):
    def __init__(self, path):
        self._features_constructor = FeaturesConstructor()
        self._features = []
        self._labels = []
        self._emotions = []
        audiopath = path + "/WAV"
        for root, dirs, files in os.walk(audiopath):
            for file in files:
                if file.endswith(".wav"):
                    audio = os.path.join(root, file)
                    ctr = audio.replace("WAV", "VALID_CTR")
                    ctr = ctr.replace(".wav", ".txt")
                    if os.path.exists(ctr):
                        emotion = emotion_id[file[0:3]]
                        feature_chunks, label_chunks = self._features_constructor.train_run(audio, ctr)
                        for chunk in feature_chunks:
                            chunk.append(np.array([emotion]))
                        self._features.extend(feature_chunks)
                        self._labels.extend(label_chunks)
        self._file_cnt = len(self._features)

    def __len__(self):
        return self._file_cnt

    def __getitem__(self, idx):
        return self._features[idx], self._labels[idx]


def getDataLoader(train_path, valid_path):
    train_dataset = FeatureDataset(train_path)
    valid_dataset = FeatureDataset(valid_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    return train_loader, valid_loader


if __name__ == "__main__":
    train, valid = getDataLoader("/PATH_TO_DATASET/train", "/PATH_TO_DATASET/validation")
    print(len(train))
    print(len(valid))
    torch.save(train, "train_loader.pth")
    torch.save(valid, "valid_loader.pth")