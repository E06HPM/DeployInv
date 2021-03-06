# Generated by Django 3.1.2 on 2021-06-24 00:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0004_auto_20210623_0415'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelSave',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(auto_now=True)),
                ('model_name', models.CharField(max_length=50)),
                ('model_mape', models.DecimalField(decimal_places=2, max_digits=5)),
            ],
        ),
    ]
