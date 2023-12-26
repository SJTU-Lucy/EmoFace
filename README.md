# EmoFace: Audio-driven Emotional 3D Face Animation

PyTorch implementation for the paper **EmoFace: Audio-driven Emotional 3D Face Animation**.

### Environment

- Linux
- Python 3.7.0
- Pytorch 1.9.0
- CUDA 11.3

### Data

As we could not publish the full dataset yet, we provide one piece of data from the evaluation set. 

- **ACTOR**: Original performance video of the actor.
- **VALID_CTR**: Extracted controller value sequence.
- **VIDEO**: MetaHuman Animation generated by **VALID_CTR**
- **WAV**: Audio clip of the corresponding video.

### Training and Testing

First, dataloader for training and testing need to be generated by *data.py*. The path to the dataset needs to be assigned.

Training and Testing is combined in *main.py*. To run the model with default settings, there is only need to set maximum epoches.

```
 python main.py --max_epoch 1000
```

When training, the weight would be saved in directory **weight/**.

### Demo

*demo.py* uses the trained model to output corresponding controller rig values for audio clips. The weight of model **PATH**,  path to audio files **audio_path** and save files **pred_path** need to be assigned inside the script.   

### Visualization

Output of the model is *.txt* files containing controller values, each row stands for one frame. To visualize the output, you need a MetaHuman model with all the controller rigs in the *valid_attr_names.txt*.