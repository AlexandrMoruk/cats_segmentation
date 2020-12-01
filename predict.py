import argparse
from model import DeepLab
import torch
from catalyst.utils import imread
import torch.nn.functional as F
import albumentations as albu
from albumentations.pytorch import ToTensor
import numpy as np
from pathlib import PurePath
from skimage import io
import PIL
from list_classes import palete

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def save_image(image, mask, path):
    mask_path = f"{PurePath(path).parent}/{PurePath(path).stem}_mask.png"
    colorize_mask(mask, palete).save(mask_path)
    mask = imread(mask_path)
    out_im = image // 2 + mask // 2
    io.imsave(path, out_im)
    return mask_path

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLab(num_classes=18, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    image = imread(args.image_path)
    valid_transformation = albu.Compose([albu.Normalize(), ToTensor()])
    im = valid_transformation(image=image)["image"].unsqueeze(0)
    prediction = model(im.to(device))
    prediction = prediction.squeeze(0).detach().cpu().numpy()

    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
    save_image(image, prediction, args.out_path)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Cat segmentation prediction')
    parser.add_argument('--model-path', metavar='DIR', help='path to model', default='model/best.pth')
    parser.add_argument('--image-path', metavar='DIR', help='path to input image', default='test_images/700gc.jpg')
    parser.add_argument('--out-path', metavar='DIR', help='path to output image', default='output.jpg')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()





