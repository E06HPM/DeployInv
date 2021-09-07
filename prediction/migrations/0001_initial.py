# Generated by Django 3.1.2 on 2021-06-27 16:58

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
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
        migrations.CreateModel(
            name='Rawdata',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(unique=True)),
                ('contract_amount', models.DecimalField(decimal_places=2, max_digits=18)),
                ('contract_qty', models.DecimalField(decimal_places=2, max_digits=10)),
                ('requisition_amount', models.DecimalField(decimal_places=2, max_digits=18)),
                ('requisition_qty', models.DecimalField(decimal_places=2, max_digits=10)),
                ('request_amount', models.DecimalField(decimal_places=2, max_digits=18)),
                ('request_qty', models.DecimalField(decimal_places=2, max_digits=10)),
                ('receive_amount', models.DecimalField(decimal_places=2, max_digits=18)),
                ('receive_qty', models.DecimalField(decimal_places=2, max_digits=10)),
                ('provide_amount', models.DecimalField(decimal_places=2, max_digits=18)),
                ('RJCN', models.DecimalField(decimal_places=0, max_digits=10)),
                ('LME_aluminum_price', models.DecimalField(decimal_places=2, max_digits=18)),
                ('revenue', models.DecimalField(decimal_places=2, max_digits=18)),
                ('scrapped_amount', models.DecimalField(decimal_places=2, max_digits=18)),
                ('IR_amount', models.DecimalField(decimal_places=2, max_digits=18)),
                ('IR_qty', models.DecimalField(decimal_places=0, max_digits=10)),
                ('lead_time', models.IntegerField()),
                ('supplier_qty', models.DecimalField(decimal_places=0, max_digits=10)),
                ('YoY_revenue', models.DecimalField(decimal_places=2, max_digits=10)),
                ('LME_nickel_price', models.DecimalField(decimal_places=2, max_digits=18)),
                ('outsource_working_hr', models.DecimalField(decimal_places=2, max_digits=18, null=True)),
                ('PER', models.DecimalField(decimal_places=2, max_digits=10, null=True)),
                ('PBR', models.DecimalField(decimal_places=2, max_digits=10, null=True)),
                ('total_csat', models.IntegerField(null=True)),
                ('raw_mat_cost_wo', models.DecimalField(decimal_places=0, max_digits=18)),
                ('semi_fin_prod_cost_wo', models.DecimalField(decimal_places=0, max_digits=18)),
                ('wip_prod_cost_wo', models.DecimalField(decimal_places=0, max_digits=18)),
                ('invt_cost_wo', models.DecimalField(decimal_places=0, max_digits=18)),
                ('raw_mat_cost', models.DecimalField(decimal_places=0, max_digits=18)),
                ('semi_fin_prod_cost', models.DecimalField(decimal_places=0, max_digits=18)),
                ('wip_prod_cost', models.DecimalField(decimal_places=0, max_digits=18)),
                ('invt_cost', models.DecimalField(decimal_places=0, max_digits=18)),
            ],
        ),
        migrations.CreateModel(
            name='CustomerSat',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('customer', models.CharField(max_length=50)),
                ('csat', models.DecimalField(decimal_places=2, max_digits=5)),
                ('date', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='prediction.rawdata')),
            ],
        ),
    ]
