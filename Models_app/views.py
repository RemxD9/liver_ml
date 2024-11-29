from PIL import Image
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
import logging
import cv2

logger = logging.getLogger('Views')


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
        self.model = self.load_model(device)
        self.data = self.load_data()
        self.flag = self.prepare()

    def prepare(self):
        if not self.model:
            self.load_model(self.device)
        data = self.load_data()
        tqdm(self.show_result(image=data))
        return True

    def load_model(self, device):
        checkpoint = torch.load('staticfiles/model/unet_liver_512_089.model')
        model = UNET()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return self.model

    def load_data(self):
        return load(self.orig_path)

    def show_result(self, image):
        fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(10, 5))
        pred = (self.model(image.to(self.device).float().unsqueeze(0))
                .squeeze(0).cpu().detach() > 0.9)
        pred_8uc1 = (pred.numpy().squeeze() * 255).astype(np.uint8)
        contours_pred, _ = cv2.findContours(pred_8uc1,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        # Пока так, подумаю как это распараллелить
        ax1.imsave(f'staticfiles/output/original_1.png', image.squeeze(), cmap='gray')
        ax2.imshow(image.squeeze(), cmap='gray')
        ax2.imshow(pred.squeeze(), alpha=0.5, cmap='autumn')
        for contour in contours_pred:
            ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)
        ax2.imsave(f'staticfiles/output/prediction_1.png')


def main_page(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            logger.info('Form is valid! Starting reading NiFTI file')
            nifti_file = form.cleaned_data['nifti_file']
            if nifti_file:
                file_path = 'staticfiles/input/uploaded_image.nii'
                with open(file_path, 'wb') as f:
                    for chunk in nifti_file.chunks():
                        f.write(chunk)
                logger.info('Successfully saved NiFTI file! Preparing to show it.')
                mean, std = (-485.18832664531345, 492.3121911082333)
                # img_3d = nib.load(file_path).get_fdata().transpose(2, 0, 1)
                trans = A.Compose([A.Resize(height=512, width=512)])
                ans = trans(image=(load(file_path)[0] - mean) / std)
                img_3d = ans['image']
                # for idx, img_slice in enumerate(img_3d):
                # img_feature = np.clip(img_slice, 0, 1)
                # transformed = trans(image=img_feature)
                # img_feature_ = transformed['image'].squeeze(0).float().cpu().numpy()
                # img_feature_ = (255 * (img_feature_ - img_feature_.min()) / (
                # img_feature_.max() - img_feature_.min())).astype(np.uint8)
                output_dir = 'staticfiles/input/'
                output_image_pil = Image.fromarray(img_3d)
                output_image_path = os.path.join(output_dir, f'uploaded_image.png')
                output_image_pil.save(output_image_path)
                logger.info('Successfully saved NiFTI file. Redirecting...')
                return redirect('watching_photos')
        else:
            logger.warn("You've most likely chosen a wrong file. Try again!")
            return render(request, 'mainpage.html', {'form': form})
    else:
        logger.warn("You've most likely chosen a file with not right format. Try again!")
        form = UploadForm()
    return render(request, 'mainpage.html', {'form': form})


def watching_photos(request):
    image_path = 'input/uploaded_image.png'
    logger.info('Redirected successfully!')
    return render(request, 'watching.html',
                  context={'image_path': image_path})


def predict(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model(device=device, image_path='staticfiles/input/uploaded_image.nii')
    logger.info('Redirecting to results')
    return redirect('results')


def results(request):
    output_image_path = 'outputs/pred.png'
    shape = request.session.get('shape')
    return render(request, 'result.html',
                  context={'output_image_path': output_image_path,
                           'shape': int(shape) - 1})


def clear_input(request):
    input_dir = 'staticfiles/input/'
    try:
        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            logger.info('Cleaning images')
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
            logger.info('Cleaning images')
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        pass
    return redirect('main')
