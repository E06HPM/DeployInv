# Generated by Django 3.1.2 on 2021-06-23 03:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0002_auto_20210616_0759'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rawdata',
            name='date',
            field=models.DateField(),
        ),
    ]