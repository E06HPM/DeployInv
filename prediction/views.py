import warnings
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.contrib import messages, auth
from prediction import models

import json
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import *

from pathlib import Path

from . import algorithm_kernel as ai
from . import data_processing as dp

# Create your views here.
warnings.simplefilter(action='ignore', category=FutureWarning)
BASE_DIR = Path(__file__).resolve().parent.parent


def login(request):
    if request.method == 'POST':
        urid = request.POST['user_id'].strip()
        urpwd = request.POST['user_pwd']
        if urid == "" or urpwd == "":
            messages.add_message(request, messages.INFO, '請檢查輸入的內容')
        else:
            try:
                user = authenticate(username=urid, password=urpwd)
                if user is not None:
                    auth.login(request, user)
                    messages.add_message(request, messages.SUCCESS, '成功登入')
                    username = user.username

                    return render(request, 'index.html', locals())
                    # return redirect('/',locals())
                else:
                    messages.add_message(request, messages.WARNING, '密碼錯誤，請再檢查')
            except:
                messages.add_message(request, messages.WARNING, '找不到使用者')

    return render(request, 'login.html', locals())


#
# @login_required(login_url='/login/')
def logout(request):
    auth.logout(request)
    messages.add_message(request, messages.INFO, '成功登出')
    return render(request, 'login.html', locals())


def predict(request):
    if request.user.is_authenticated:

        params = {
            'cost_type': 'raw_mat_cost_wo',
            'period_type': '0',
            'algorithm_type': '0',
            'start_date': '',
            'end_date': '',
            'date_input': '',
            'revenue': '',
            'requisition_qty': '',
            'request_qty': '',
            'scrapped_amount': '',
            'request_amount': ''
        }

        username = request.user.username
        ref_cost = {'raw_mat_cost_wo': '原物料存貨金額(不含WRITE-OFF)',
                    'semi_fin_prod_cost_wo': '半製品存貨金額(不含WRITE-OFF)',
                    'wip_prod_cost_wo': '在製品存貨金額(不含WRITE-OFF)',
                    'invt_cost_wo': '總存貨金額(不含WRITE-OFF)',
                    'raw_mat_cost': '原物料存貨金額',
                    'semi_fin_prod_cost': '半製品存貨金額',
                    'wip_prod_cost': '在製品存貨金額',
                    'invt_cost': '總存貨金額',
                    'revenue':'營收'}
        post_training = False
        rawdata = models.Rawdata.objects.all()
        date_df = pd.DataFrame(rawdata.values('date'))
        min_date = date_df['date'].min()

        max_date = date_df['date'].max()
        current_date = datetime.now()
        str_min_data = str(max_date.year) + '-' + str(max_date.month)
        latest_data = datetime.now() + relativedelta(months=6)
        str_latest_data = str(latest_data.year) + '-' + str(latest_data.month)

        # try:

        if request.method == 'POST':
            period_type = request.POST['name_period']
            post_training = True

            params['period_type'] = period_type

            if period_type == '0':  # multiple

                date_max = False
                date_min = False
                algorithm_type = request.POST['name_algorithm']
                params['algorithm_type'] = algorithm_type
                date_start = request.POST['name_date_start']
                date_end = request.POST['name_date_end']

                if date_start == '' or date_end == '':
                    post_training = False
                    param = json.dumps(params)
                    messages.add_message(request, messages.WARNING, '填入資料不正確')
                    return render(request, 'predict.html', locals())
                else:
                    date_start = date_start + '-01'
                    date_end = date_end + '-01'
                    date_start = datetime.strptime(str(date_start), "%Y-%m-%d").date()
                    date_end = datetime.strptime(str(date_end), "%Y-%m-%d").date()
                    str_min_data = str(date_start.year) + '-' + str(date_start.month)
                    str_latest_data = str(date_end.year) + '-' + str(date_end.month)
                    rawdata = models.Rawdata.objects.all()
                    date_min = min_date > date_start

                    time_start = str(min_date)  # 輸入資料起始時間
                    point_time_start = time_start.replace('-', '/')
                    time_end = str(max_date)  # 輸入資料結束時間
                    predict_start = str(date_start)  # 決定訓練集要起始時間
                    predcit_end = str(date_end)  # 決定訓練集要結束時間
                    orig_year = min_date.year
                    orig_month = min_date.month
                    pred_year = date_start.year
                    pred_month = date_start.month

                    result_pred = dict()

                    for i in ref_cost.keys():
                        print(i)
                        cost_type = i
                        data = pd.DataFrame(rawdata.values('date', cost_type))
                        data = dp.history_data_covert(data, cost_type)
                        pred_obj = ai.ModelLoadPred(data, cost_type, period_type, algorithm_type, predict_start,
                                                    predcit_end)
                        pred_obj = pred_obj.load_pred()
                        pred_obj = pred_obj.squeeze()


                        if cost_type == 'revenue':
                            
                            result_pred[i] = dp.data_for_ploting(pred_obj.cumsum(axis=0))
                        else:
                            result_pred[i] = dp.data_for_ploting(pred_obj)
                            result_pred['orig_' + i] = dp.data_for_ploting(data)


            elif period_type == '1':

                single_output = {'raw_mat_cost_wo': 0,
                                    'semi_fin_prod_cost_wo': 0,
                                    'wip_prod_cost_wo': 0,
                                    'invt_cost_wo': 0,
                                    'raw_mat_cost': 0,
                                    'semi_fin_prod_cost': 0,
                                    'wip_prod_cost': 0,
                                    'invt_cost': 0,
                                    'revenue': 0
                                    }

                period_type = request.POST['name_period']
                params['period_type'] = period_type
                algorithm_type = request.POST['name_algorithm']
                params['algorithm_type'] = algorithm_type

                params['date_input'] = request.POST['name_date_input']
                params['requisition_qty'] = request.POST['name_requisition_qty']
                params['request_qty'] = request.POST['name_request_qty']
                params['scrapped_amount'] = request.POST['name_scrapped_amount']
                params['request_amount'] = request.POST['name_request_amount']

                x0 = request.POST['name_date_input'] + '-01'
                x2 = int(request.POST['name_requisition_qty'])
                x3 = int(request.POST['name_request_qty'])
                x4 = int(request.POST['name_scrapped_amount'])
                x5 = int(request.POST['name_request_amount'])

                datecoder = pd.DataFrame(models.DateEncoder.objects.all().values())
                x0 = int(datecoder[datecoder['date'] == pd.to_datetime(x0)]['encoder'].values)
                data = np.array([x0, x2, x3, x4, x5])
                data = data.reshape(1, 5)

                for i in single_output.keys():
                    cost_type = i
                    pred_obj = ai.ModelLoadPred(data, cost_type, period_type, '0', x0, x0)
                    single_output[cost_type] = int(pred_obj.load_pred())

                single_output = json.dumps(single_output)

        param = json.dumps(params)

        # except:
        #     post_training = False
        #     param = json.dumps(params)
        #     messages.add_message(request, messages.WARNING, '填入資料不正確')
        #     return render(request, 'predict.html', locals())


    return render(request, 'predict.html', locals())


def training(request):
    if request.user.is_authenticated:

        params = {
            'cost_type': 'raw_mat_cost_wo',
            'period_type': '0',
            'algorithm_type': '0',
            'start_date': '',
            'end_date': ''
        }

        username = request.user.username
        ref_cost = {'raw_mat_cost_wo': '原物料存貨金額(不含WRITE-OFF)',
                    'semi_fin_prod_cost_wo': '半製品存貨金額(不含WRITE-OFF)',
                    'wip_prod_cost_wo': '在製品存貨金額(不含WRITE-OFF)',
                    'invt_cost_wo': '總存貨金額(不含WRITE-OFF)',
                    'raw_mat_cost': '原物料存貨金額',
                    'semi_fin_prod_cost': '半製品存貨金額',
                    'wip_prod_cost': '在製品存貨金額',
                    'invt_cost': '總存貨金額',
                    'revenue':'營收'}

        post_training = False
        rawdata = models.Rawdata.objects.all()
        date_df = pd.DataFrame(rawdata.values('date'))
        min_date = date_df['date'].min()
        str_min_data = str(min_date)
        max_date = date_df['date'].max()

        year1 = datetime.strptime(str(max_date), "%Y-%m-%d").year
        year2 = datetime.strptime(str(min_date), "%Y-%m-%d").year
        month1 = datetime.strptime(str(max_date), "%Y-%m-%d").month
        month2 = datetime.strptime(str(min_date), "%Y-%m-%d").month
        num = int(((year1 - year2) * 12 + (month1 - month2)) * .9)

        training_end = min_date + relativedelta(months=num)

        str_min_data = str(min_date.year) + '-' + str(min_date.month)
        str_latest_data = str(training_end.year) + '-' + str(training_end.month)
        params['start_date'] = str_min_data
        params['end_date'] = str_latest_data

        try:


            if request.method == 'POST':

                period_type = request.POST['name_period']
                cost_type = request.POST['name_cost']
                algorithm_type = request.POST['name_algorithm']

                if request.POST.get("training"):
                    period_type = request.POST['name_period']
                    cost_type = request.POST['name_cost']
                    algorithm_type = request.POST['name_algorithm']
                    params['period_type'] = period_type
                    params['cost_type'] = cost_type
                    params['algorithm_type'] = algorithm_type

                    if period_type == '0':  # multiple
                        # try:

                        date_max = False
                        date_min = False

                        date_start = request.POST['name_date_start']
                        date_end = request.POST['name_date_end']
                        params['start_date'] = date_start
                        params['end_date'] = date_end

                        if date_start == '' or date_end == '':
                            post_training = False
                            param = json.dumps(params)
                            messages.add_message(request, messages.WARNING, '填入資料不正確')
                            return render(request, 'training.html', locals())
                        else:

                            date_start = date_start + '-01'
                            date_end = date_end + '-01'
                            date_start = datetime.strptime(str(date_start), "%Y-%m-%d").date()
                            date_end = datetime.strptime(str(date_end), "%Y-%m-%d").date()

                            rawdata = models.Rawdata.objects.all()
                            data = pd.DataFrame(rawdata.values(cost_type))

                            date_max = max_date < date_end
                            date_min = min_date > date_start

                            if date_max:
                                messages.add_message(request, messages.WARNING, '結束日期超過資料庫最大日期：' + str(max_date))
                                return render(request, 'training.html', locals())
                            if date_min:
                                messages.add_message(request, messages.WARNING, '開始日期超過資料庫最小日期：' + str(min_date))
                                return render(request, 'training.html', locals())

                            time_start = str(min_date)  # 輸入資料起始時間
                            point_time_start = time_start.replace('-', '/')
                            time_end = str(max_date)  # 輸入資料結束時間
                            train_start = str(date_start)  # 決定訓練集要起始時間
                            train_end = str(date_end)  # 決定訓練集要結束時間
                            train_year = date_start.year
                            train_month = date_start.month
                            predict_start = (date_end + relativedelta(months=1))  # 決定測試集起始時間，也就是要預測區間的起始時間
                            pred_year = predict_start.year

                            pred_month = predict_start.month
                            predcit_end = str(max_date)  # 決定測試集結束時間，也就是要預測區間的結束時間
                            cost_type = cost_type  # 決定要預測哪一類的存貨金額：原物料存貨金額(不含WRITE-OFF)、半製品存貨金額(不含WRITE-OFF)、在製品存貨金額(不含WRITE-OFF)、總存貨金額(不含WRITE-OFF)、原物料存貨金額、半製存貨金額、在製存貨金額、總存貨金額
                            # 建立result物件，用以分析存貨成本的資料模式屬於序列型態，以決定採用的模型（函數名稱）

                            if algorithm_type == '0':

                                result_new = ai.TimeSeriesInventory(data, cost_type, time_start, time_end, train_start,
                                                                    train_end,
                                                                    predict_start, predcit_end)

                                result_new.data_split()
                                training_data, testing_data, prediction_data, mape = result_new.sarima()
                                prediction_data = prediction_data.squeeze()

                                mape = round(mape, 2)
                                training_dataset = dp.data_for_ploting(training_data)
                                testing_dataset = dp.data_for_ploting(testing_data)
                                prediction_dateset = dp.data_for_ploting(prediction_data)

                            elif algorithm_type == '1':

                                result_new = ai.TimeSeriesInventory(data, cost_type, time_start, time_end, train_start,
                                                                    train_end,
                                                                    predict_start, predcit_end)
                                # #依據下圖結果，對應到圖一，可判斷此筆成本屬於趨勢型，故可採用arima、sarima、holt、holt_winters等模型
                                result_new.data_split()
                                training_data, testing_data, prediction_data, mape = result_new.holt_winters()
                                prediction_data = prediction_data.squeeze()

                                mape = round(mape, 2)

                                training_dataset = dp.data_for_ploting(training_data)
                                testing_dataset = dp.data_for_ploting(testing_data)
                                prediction_dateset = dp.data_for_ploting(prediction_data)

                            if algorithm_type == '2':
                                result_new = ai.LSTMInventory(data, cost_type, time_start, time_end, train_start, train_end,
                                                              predict_start, predcit_end)
                                result_new.data_preprocessing()
                                training_data, testing_data, prediction_data, mape = result_new.lstm()
                                prediction_data = prediction_data.squeeze()

                                mape = round(mape, 2)

                                training_dataset = dp.data_for_ploting(training_data)
                                testing_dataset = dp.data_for_ploting(testing_data)
                                prediction_dateset = dp.data_for_ploting(prediction_data)

                    else:  # single

                        if cost_type =='revenue':
                            featurs = ['date', 'requisition_qty', 'request_qty', 'scrapped_amount',
                                       'request_amount']
                        else:
                            featurs = ['date', 'requisition_qty', 'request_qty', 'scrapped_amount',
                                       'request_amount']


                        # featurs = ['date', 'contract_amount', 'contract_qty', 'requisition_amount',
                        #            'requisition_qty', 'request_amount', 'request_qty', 'receive_amount',
                        #            'receive_qty', 'provide_amount', 'RJCN', 'LME_aluminum_price',
                        #            'revenue', 'scrapped_amount', 'IR_amount', 'IR_qty', 'lead_time',
                        #            'supplier_qty', 'YoY_revenue', 'LME_nickel_price']

                        rawdata = pd.DataFrame(rawdata.values())
                        rawdata['date'] = pd.to_datetime(rawdata['date'])

                        date_encoder = pd.DataFrame(models.DateEncoder.objects.all().values())

                        date_encoder['date'] = pd.to_datetime(date_encoder['date'])

                        rawdata = rawdata.merge(date_encoder[['date', 'encoder']], left_on='date', right_on='date')
                        rawdata = rawdata.set_index('date')
                        rawdata.rename(columns={'encoder': 'date'}, inplace=True)
                        print(rawdata)

                        y = rawdata[cost_type]
                        X = rawdata[featurs]
                        print(X)

                        if algorithm_type == '0':

                            pred = ai.MachineLearning(X, y, cost_type, featurs)
                            training_data, prediction_data, mape = pred.randomForest()
                            prediction_data = prediction_data.squeeze()

                            mape = round(mape, 2)
                            training_dataset = dp.data_for_ploting(training_data)
                            testing_dataset = 0
                            prediction_dateset = dp.data_for_ploting(prediction_data)

                        # elif algorithm_type == '1':
                        #     pred = ai.MachineLearning(X, y, cost_type, featurs)
                        #     training_data, prediction_data, mape = pred.MARS()
                        #     prediction_data = prediction_data.squeeze()

                        #     mape = round(mape, 2)
                        #     training_dataset = dp.data_for_ploting(training_data)
                        #     testing_dataset = 0
                        #     prediction_dateset = dp.data_for_ploting(prediction_data)

                    post_training = True

                elif request.POST.get("saving"):

                    post_training = False
                    cost_type = request.POST['name_cost']
                    algorithm_type = request.POST['name_algorithm']
                    period_type = request.POST['name_period']
                    model = ai.ModelSaving(cost_type, algorithm_type, period_type)
                    model.modelSave()
                    params['period_type'] = period_type
                    params['cost_type'] = cost_type
                    params['algorithm_type'] = algorithm_type

            param = json.dumps(params)

        except:
            post_training = False
            param = json.dumps(params)
            messages.add_message(request, messages.WARNING, '填入資料不正確')
            return render(request, 'training.html', locals())

    return render(request, 'training.html', locals())
