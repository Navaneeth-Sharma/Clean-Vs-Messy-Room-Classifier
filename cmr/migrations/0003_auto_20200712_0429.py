# Generated by Django 3.0.7 on 2020-07-12 04:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cmr', '0002_image_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='img',
            field=models.ImageField(upload_to='media/'),
        ),
        migrations.AlterField(
            model_name='image',
            name='name',
            field=models.CharField(default='2020-07-12 04:29:31.491291', max_length=50),
        ),
    ]
