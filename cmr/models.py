from django.db import models
from datetime import datetime

# Create your models here.

class Image(models.Model):
    name = models.CharField(max_length=50)
    img = models.ImageField(upload_to="media/",default='')

    def __str__(self):
        return self.name

    # def image_tag(self):
    #     from django.utils.html import escape
    #     from django.conf import settings
    #     return u'<img src="%s" />' % escape(settings.MEDIA_URL)
    # image_tag.short_description = 'Image'
    # image_tag.allow_tags = True

