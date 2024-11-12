from PIL import Image
import numpy as np
import torch
from django.shortcuts import redirect, render
import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib
from Models_app.forms import UploadForm
import os
import logging


logger = logging.getLogger('Views')


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
                img_3d = nib.load(file_path).get_fdata().transpose(2, 0, 1)
                shape = img_3d.shape[0]
                request.session['shape'] = shape
                trans = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
                for idx, img_slice in enumerate(img_3d):
                    img_feature = np.clip(img_slice, 0, 1)
                    transformed = trans(image=img_feature)
                    img_feature_ = transformed['image'].squeeze(0).float().cpu().numpy()
                    img_feature_ = (255 * (img_feature_ - img_feature_.min()) / (
                                img_feature_.max() - img_feature_.min())).astype(np.uint8)
                    output_dir = 'staticfiles/input/'
                    output_image_pil = Image.fromarray(img_feature_)
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
    return render(request, 'watching.html', context={'image_path': image_path})


def predict(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Initiating chosen model, please wait')
    model = load_model(device=device)
    logger.info(f'Model {type(model).__name__} initialized successfully')
    image_path = 'staticfiles/input/uploaded_image.nii'
    logger.info('Preparing slices for model, please wait')
    data = load_data(image_path=image_path)
    logger.info('Preparing graphs, please wait')
    seg_graphs(model=model, slices=data, device=device)
    logger.info('Redirecting to results')

    return redirect('results')


def results(request):
    output_image_path = 'outputs/pred.png'
    shape = request.session.get('shape')
    return render(request, 'result.html', context={'output_image_path': output_image_path,
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
