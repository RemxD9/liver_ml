from django import forms
from django.core.exceptions import ValidationError


class UploadForm(forms.Form):
    dcm_file = forms.FileField(label='Select a DCM file', required=False,
                               widget=forms.ClearableFileInput(attrs={'class': 'button-load'}))

    def clean_nifti_file(self):
        file = self.cleaned_data.get('dcm_file')
        if file:
            if not file.name.endswith('.dcm'):
                raise ValidationError('Неверный формат файла. Ожидается файл с расширением .nii')
        return file
