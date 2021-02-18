# CNN-FACIAN-EXPRESSION

Detecting face emotion with OpenCV and TensorFlow. Using simple CNN or model provided by TensorFlow as MobileNetV2, VGG16, Xception.

## Data

Raw data collected from kaggle with `fer2013`+`ferplus for labels` ,

## Training

Execute `train.py` script and pass network architecture type dataset dir and epochs to it.
Default network type is MobileNetV2.

## Model result

| Model         | Test Accuracy| Test lost| 
| ------------- | -------------|------------- |
| [neha01 model](https://github.com/neha01/Realtime-Emotion-Detection)            |  **81.67%**      |  **19.46%** |
| MobileNetV2 (fine tune)  |  80.64%   | 19.84% |
| VGG16         |  78.44%      | 20.85% |
| Xception | 64.04%   |  26.45% |
