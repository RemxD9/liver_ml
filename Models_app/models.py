from django.db import models
import logging


logger = logging.getLogger('Models')


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
