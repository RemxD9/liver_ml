from django import forms
from django.core.exceptions import ValidationError
from .models import UploadedImage


class UploadForm(forms.Form):
    nifti_file = forms.FileField(label='Select a NIfTI file', required=False)

    def clean_nifti_file(self):
        file = self.cleaned_data.get('nifti_file')
        if file:
            if not file.name.endswith('.nii'):
                raise ValidationError('Неверный формат файла. Ожидается файл с расширением .nii')
        return file
