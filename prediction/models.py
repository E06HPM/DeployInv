from django.db import models
from datetime import datetime

# Create your models here.

class Rawdata(models.Model):
    date = models.DateField(unique=True)
    contract_amount = models.DecimalField(max_digits=18, decimal_places=2)
    contract_qty = models.DecimalField(max_digits=10, decimal_places=2)
    requisition_amount = models.DecimalField(max_digits=18, decimal_places=2)
    requisition_qty = models.DecimalField(max_digits=10, decimal_places=2)
    request_amount = models.DecimalField(max_digits=18, decimal_places=2)
    request_qty = models.DecimalField(max_digits=10, decimal_places=2)
    receive_amount = models.DecimalField(max_digits=18, decimal_places=2)
    receive_qty = models.DecimalField(max_digits=10, decimal_places=2)
    provide_amount = models.DecimalField(max_digits=18, decimal_places=2)
    RJCN = models.DecimalField(max_digits=10, decimal_places=0)
    LME_aluminum_price = models.DecimalField(max_digits=18, decimal_places=2)
    revenue = models.DecimalField(max_digits=18, decimal_places=2)
    scrapped_amount = models.DecimalField(max_digits=18, decimal_places=2)
    IR_amount = models.DecimalField(max_digits=18, decimal_places=2)
    IR_qty = models.DecimalField(max_digits=10, decimal_places=0)
    lead_time = models.IntegerField()
    supplier_qty = models.DecimalField(max_digits=10, decimal_places=0)
    YoY_revenue = models.DecimalField(max_digits=10, decimal_places=2)
    LME_nickel_price = models.DecimalField(max_digits=18, decimal_places=2)
    outsource_working_hr = models.DecimalField(max_digits=18, decimal_places=2,null=True)
    PER = models.DecimalField(max_digits=10, decimal_places=2,null=True)
    PBR = models.DecimalField(max_digits=10, decimal_places=2,null=True)
    total_csat = models.IntegerField(null=True)
    raw_mat_cost_wo = models.DecimalField(max_digits=18, decimal_places=0)
    semi_fin_prod_cost_wo = models.DecimalField(max_digits=18, decimal_places=0)
    wip_prod_cost_wo = models.DecimalField(max_digits=18, decimal_places=0)
    invt_cost_wo = models.DecimalField(max_digits=18, decimal_places=0)
    raw_mat_cost = models.DecimalField(max_digits=18, decimal_places=0)
    semi_fin_prod_cost = models.DecimalField(max_digits=18, decimal_places=0)
    wip_prod_cost = models.DecimalField(max_digits=18, decimal_places=0)
    invt_cost = models.DecimalField(max_digits=18, decimal_places=0)

    # def __str__(self):
    #     return str(self.date)

class CustomerSat(models.Model):
    date = models.ForeignKey(Rawdata, on_delete=models.CASCADE)
    customer = models.CharField(max_length=50)
    csat = models.DecimalField(max_digits=5, decimal_places=2)

class ModelSave(models.Model):
    date = models.DateField(auto_now=True)
    model_name = models.CharField(max_length=50)
    model_mape = models.DecimalField(max_digits=5, decimal_places=2)

class DateEncoder(models.Model):
    date = models.DateField(auto_now=True)
    encoder = models.IntegerField(unique=True)


class ModelStandardized(models.Model):
    date = models.DateField(auto_now=True)
    model_name = models.CharField(max_length=50)
    feature = models.CharField(max_length=50)
    mean = models.DecimalField(max_digits=18, decimal_places=2)
    std = models.DecimalField(max_digits=18, decimal_places=2)
    flag = models.BooleanField(default=False)
