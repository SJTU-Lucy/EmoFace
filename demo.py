import torch
import numpy as np
from net import EmoFace
from scipy import signal
import os
import random
import math
from data import FeaturesConstructor

emotion_id = {"ang": 0, "dis": 1, "fea": 2, "hap": 3, "neu": 4, "sad": 5, "sur": 6}
emo_dim = 7                                 # dimension of emotions
out_dim = 174                               # dimension of output controller value
PATH = "weights/1000_model.pth"             # path to saved .pth weight file
valid_data = "/PATH_TO_DATASET/validation"
audio_path = valid_data + "/WAV"            # path of audios to be predicted
pred_path = valid_data + "/PRED_CTR"        # path to save predicted controller values
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# blink settings
blink_dic = [0.95, 0.75, 0.55, 0.3, 0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9]
blink_dur = 13
mu = 3.518
sigma = 0.532

# gaze settings
switch_dur = 5
return_rate = 0.4
interval_lower = 25
interval_upper = 45
radius_lower = 0.1
radius_upper = 0.2

model = EmoFace(emo_dim=emo_dim, out_dim=out_dim)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

# sample interval between blinks
def sample_interval():
    sampled_frequency = random.lognormvariate(mu, sigma)
    sampled_frequency = min(100, max(15, sampled_frequency))
    interval_frame = int(3600 / sampled_frequency)
    return interval_frame

def add_blink(pred):
    interval_frame = sample_interval()
    blink_count = 0
    for i in range(pred.shape[0]):
        k = 1
        if interval_frame >= 0:
            interval_frame -= 1
        else:
            k = blink_dic[blink_count]
            blink_count += 1
            if blink_count == blink_dur:
                interval_frame = sample_interval()
                blink_count = 0
            # rig 37 and 106 stands for blinks
            pred[i][36] = 1 - (1 - pred[i][36]) * k
            pred[i][105] = 1 - (1 - pred[i][105]) * k
    return pred


def add_gaze(pred):
    gaze_interval = random.randint(25, 45)
    switch_count = 0
    current_point = (0, 0)
    next_point = None
    for i in range(pred.shape[0]):
        if gaze_interval >= 0:
            gaze_interval -= 1
            coordinate = current_point
        else:
            if switch_count == 0:
                if random.random() < 0.4:  # 40% chance to return to (0, 0)
                    next_point = (0, 0)
                else:
                    radius = random.uniform(radius_lower, radius_upper)
                    theta = random.uniform(0, 2 * math.pi)
                    next_point = (radius * math.cos(theta), radius * math.sin(theta))
                switch_count = 1
            x = current_point[0] + (next_point[0] - current_point[0]) * switch_count / switch_dur
            y = current_point[1] + (next_point[1] - current_point[1]) * switch_count / switch_dur
            coordinate = (x, y)
            switch_count += 1
            if switch_count > switch_dur:
                gaze_interval = random.randint(25, 45)
                switch_count = 0
        # the first and second rig stands for eye gaze
        pred[i][0] = coordinate[0]
        pred[i][1] = coordinate[1]
    return pred

def predict(audio_file, pred_file, emotion):
    feature_constructor = FeaturesConstructor()
    feature_chunks = feature_constructor.infer_run(audio_file)
    for feature_chunk in feature_chunks:
        audio, label = feature_chunk
        emotion = np.array([emotion])
        audio, label, emotion = torch.from_numpy(audio).float(), torch.from_numpy(label).float(), torch.from_numpy(emotion).int()
        audio, label, emotion = audio.unsqueeze(0), label.unsqueeze(0), emotion.unsqueeze(0)
        audio, label, emotion = audio.to(device), label.to(device), emotion.to(device)

    predict = model(audio, label, emotion)
    predict = predict.squeeze()
    result = predict.detach().cpu().numpy()

    # smooth filter
    result = result.T
    result = signal.savgol_filter(result, window_length=15, polyorder=3, mode="nearest").T
    # add blink and gaze
    result = add_blink(result)
    result = add_gaze(result)

    # write into the file
    Note = open(pred_file, mode='w')
    for i in range(len(result)):
        for j in range(out_dim):
            Note.write(str(round(result[i][j], 6)))
            if j != out_dim - 1:
                Note.write(',')
        Note.write('\n')
    Note.close()


if __name__ == '__main__':
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith(".wav"):
                emotion = emotion_id[file[0:3]]
                audio = os.path.join(root, file)
                pred = audio.replace("WAV", "PRED_CTR")
                pred = pred.replace(".wav", ".txt")
                print(audio, pred, emotion)
                predict(audio, pred, emotion)

