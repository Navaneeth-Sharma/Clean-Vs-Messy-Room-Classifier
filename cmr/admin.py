from django.contrib import admin
from cmr.models import Image

# Register your models here.
admin.site.register(Image)
fields = ( 'image_tag', )
readonly_fields = ('image_tag',)