# Korean Speaker Verification

This repository is made for the **Korean Speaker Verification task** based on "*NeXt-TDNN: Modernizing Multi-Scale Temporal Convolution Backbone for Speaker Verification*" from ICASSP 2024. [[Official Repository](https://github.com/dmlguq456/NeXt_TDNN_ASV)] [[Paper Link](https://arxiv.org/pdf/2312.08603)]

## Requirements
Install dependencies:
```
pip install -r requirements.txt
```

## Dataset

For training, the Korean Speaker Verification dataset provided by the AI hub is used. You can download the whole dataset from the link below. Registration might be required to download.

- [화자 인식용 음성 데이터 (Korean Speaker Verification Dataset)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=537)

For the augmentation, MUSAN dataset and Room Impulse Responses (RIR) dataset are used.  
- [MUSAN Dataset](https://www.openslr.org/17/)  
- [RIR Dataset](https://www.openslr.org/28/)

## Model Training & Inference
Run `main.py` for whole train/inference process:
```
python main.py
```
- Model configuration and settings are provided as a `.yaml` file in the `configs` directory.
- Please check `basic_training.ipynb` for simplified training.