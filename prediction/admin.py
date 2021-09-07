from django.contrib import admin
from .models import Rawdata, CustomerSat,ModelSave

# Register your models here.
class RawdataAdmin(admin.ModelAdmin):
    list_display = ['date']

class CustomerSatAdmin(admin.ModelAdmin):
    list_display = ['date_id','customer','csat']

class ModelSaveAdmin(admin.ModelAdmin):
    list_display = ['date','model_name','model_mape']

class ModelSaveAdmin(admin.ModelAdmin):
    list_display = ['date','model_name','model_mape']

admin.site.register(Rawdata, RawdataAdmin)
admin.site.register(CustomerSat, CustomerSatAdmin)
admin.site.register(ModelSave,ModelSaveAdmin)
