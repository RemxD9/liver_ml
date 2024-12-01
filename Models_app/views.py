from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import albumentations as A
import torch.nn as nn
from tqdm import tqdm
from medpy.io import load
import os
import numpy as np
import torch
from django.shortcuts import redirect, render
from Models_app.forms import UploadForm
import cv2


def conv_plus_conv(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
    )


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        base_channels = 8

        self.down1 = conv_plus_conv(1, base_channels)
        self.down2 = conv_plus_conv(base_channels, base_channels * 2)
        self.down3 = conv_plus_conv(base_channels * 2, base_channels * 4)
        self.down4 = conv_plus_conv(base_channels * 4, base_channels * 8)
        self.down5 = conv_plus_conv(base_channels * 8, base_channels * 16)

        self.up1 = conv_plus_conv(base_channels * 2, base_channels)
        self.up2 = conv_plus_conv(base_channels * 4, base_channels)
        self.up3 = conv_plus_conv(base_channels * 8, base_channels * 2)
        self.up4 = conv_plus_conv(base_channels * 16, base_channels * 4)
        self.up5 = conv_plus_conv(base_channels * 32, base_channels * 8)

        self.out = nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=1)

        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual1 = self.down1(x)
        x = self.downsample(residual1)

        residual2 = self.down2(x)
        x = self.downsample(residual2)

        residual3 = self.down3(x)
        x = self.downsample(residual3)

        residual4 = self.down4(x)
        x = self.downsample(residual4)

        residual5 = self.down5(x)
        x = self.downsample(residual5)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual5), dim=1)
        x = self.up5(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual4), dim=1)
        x = self.up4(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual3), dim=1)
        x = self.up3(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual2), dim=1)
        x = self.up2(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual1), dim=1)
        x = self.up1(x)

        return self.sigmoid(self.out(x))


class Model:
    def __init__(self, device, image_path):
        self.device = device
        self.orig_path = image_path
        self.model = UNET()
        self.data = self.load_data()
        self.flag = self.prepare()

    def prepare(self):
        self.load_model(self.device)
        data = self.load_data()
        tqdm(self.show_result(image=data))
        return True

    def load_model(self, device):
        checkpoint = torch.load('staticfiles/model/liver_512_089.pth', map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
        return

    def load_data(self):
        mean, std = (-485.18832664531345, 492.3121911082333)
        trans = A.Compose([A.Resize(height=512, width=512), ToTensorV2()])
        img = load(self.orig_path)[0]
        img[img > 1192] = 1192
        ans = trans(image=((img - mean) / std))
        return ans['image']

    def show_result(self, image):
        pred = (self.model
                (image.to(self.device).float().unsqueeze(0))
                .squeeze(0).cpu().detach() > 0.9)

        plt.imsave('staticfiles/output/original.png', image.squeeze(), cmap='gray')

        fig = plt.figure(frameon=False)
        fig.set_size_inches(1, 1)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        image = image.squeeze(0)
        pred_8uc1 = (pred.numpy().squeeze() * 255).astype(np.uint8)
        contours_pred, _ = cv2.findContours(pred_8uc1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ax.imshow(image.squeeze(), cmap='gray')
        ax.imshow(pred.squeeze(), alpha=0.5, cmap='gray')
        for contour in contours_pred:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=1)
        fig.savefig('staticfiles/output/prediction.png', dpi=512)


def main_page(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            dcm_file = form.cleaned_data['dcm_file']
            if dcm_file:
                file_path = 'staticfiles/input/uploaded_image.dcm'
                with open(file_path, 'wb') as f:
                    for chunk in dcm_file.chunks():
                        f.write(chunk)
                mean, std = (-485.18832664531345, 492.3121911082333)
                trans = A.Compose([A.Resize(height=512, width=512), ToTensorV2()])
                img = load(file_path)[0]
                img[img > 1192] = 1192
                ans = trans(image=((img - mean) / std))
                img_3d = ans['image']
                output_dir = 'staticfiles/input/'
                output_image_path = os.path.join(output_dir, f'uploaded_image.png')
                plt.imsave(output_image_path, img_3d[0], cmap='gray')
                return redirect('watching_photos')
        else:
            return render(request, 'mainpage.html', {'form': form})
    else:
        form = UploadForm()
    return render(request, 'mainpage.html', {'form': form})


def watching_photos(request):
    image_path = 'input/uploaded_image.png'
    return render(request, 'watching.html',
                  context={'image_path': image_path})


def predict(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model(device=device, image_path='staticfiles/input/uploaded_image.dcm')
    return redirect('results')


def results(request):
    output_pred_path = '/output/prediction.png'
    output_image_path = '/output/original.png'
    return render(request, 'result.html',
                  context={'output_pred': output_pred_path,
                           'output_orig': output_image_path})


def clear_input(request):
    input_dir = 'staticfiles/input/'
    try:
        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        pass
    return redirect('clear_output')


def clear_output(request):
    output_dir = 'staticfiles/outputs/'
    try:
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        pass
    return redirect('main')
