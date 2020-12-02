# Cats Segmentation


## Installing

```
python 3
```
#### if you have cuda 10.2 install as described below, else choose your [specifications](https://pytorch.org/get-started/locally/) 
```
pip install torch torchvision 
```

```
pip install -r requirements.txt
```

####Usage
```
python predict.py --model-path path/to/model.pth
--image-path /path/to/image.jpg 
--out-path ~/path/to/out_image.jpg
```
####Model
DeepLabV3+ model with a pre-trained ResNet-50 backbone

[Download](https://drive.google.com/drive/folders/1rWRU9vOxrMFteM0ebc7LTW7oluRCiPoa?usp=sharing)








