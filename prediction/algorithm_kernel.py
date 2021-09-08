import warnings
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import *
from shutil import move

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers

from statsmodels.tsa.api import ExponentialSmoothing, SARIMAX
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pyearth
import joblib

from .models import ModelSave, ModelStandardized

warnings.simplefilter(action='ignore', category=FutureWarning)
BASE_DIR = Path(__file__).resolve().parent.parent

class TimeSeriesInventory:

    def __init__(self, data, cost_type, time_start, time_end, train_start, train_end, predict_start, predcit_end):

        self.data = data
        self.time_interval = pd.date_range(start=time_start, end=time_end, freq='MS')  # generate the timestamp'
        self.data = self.data.set_index(self.time_interval)  # set the timestamp as a index
        self.data_timeseries = (self.data[cost_type])  # select the cost type from dataframe
        self.cost_type = cost_type
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.predict_start = pd.to_datetime(predict_start)
        self.predcit_end = pd.to_datetime(predcit_end)
        self.predcit_interval = pd.date_range(start=predict_start, end=predcit_end, freq='MS')

    # Split the original data into training & testing set
    def data_split(self):
        self.training_set = self.data_timeseries[(self.data_timeseries.index <= self.train_end)]
        self.training_set = self.training_set[(self.training_set.index >= self.train_start)]
        self.testing_set = self.data_timeseries[(self.data_timeseries.index > self.train_end)]

    def mape(self, actual, pred):
        return np.mean(np.abs((actual - pred) / actual)) * 100

    # SARIMA method
    def sarima(self):
        method_name = 'SARIMA'
        fit1 = SARIMAX((self.training_set.astype(float)), order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
        y_hat = pd.Series(fit1.predict(start=self.predict_start, end=self.predcit_end, dynamic=True).values,
                             index=self.predcit_interval)
        mape_sarima = self.mape(np.array(self.testing_set.astype(float)), np.array(y_hat.astype(float)))
        mape_sarima = round(mape_sarima, 2)


        file_names = [fn for fn in os.listdir(os.path.join(BASE_DIR, "media"))]
        for i in file_names:
            if 'temp_' in i:
                os.remove(os.path.join(BASE_DIR, "media") + '/' + i)

        model_path = os.path.join(BASE_DIR, "media") + '/temp_' + str(self.cost_type) + '-' + str(method_name)
        model_name = str(self.cost_type) + '-' + str(method_name)
        joblib.dump(fit1, model_path)

        if ModelSave.objects.filter(model_name=model_name).exists():
            mdl = ModelSave.objects.filter(model_name=model_name)
            mdl.update(model_mape=mape_sarima)
        else:
            mdl = ModelSave.objects.create(model_name=model_name, model_mape=mape_sarima)
            mdl.save()

        return self.training_set.astype(float), self.testing_set.astype(float), y_hat, mape_sarima

    # Holt Winters method
    def holt_winters(self):
        method_name = 'Holt_winters'
        fit1 = ExponentialSmoothing(np.asarray(self.training_set.astype(float)), seasonal_periods=12, trend='add',
                                    seasonal='add', ).fit()

        y_hat = pd.Series(fit1.forecast(len(self.predcit_interval)), index=self.predcit_interval)

        mape_hw = self.mape(np.array(self.testing_set.astype(float)), np.array(y_hat.astype(float)))
        mape_hw = round(mape_hw, 2)

        file_names = [fn for fn in os.listdir(os.path.join(BASE_DIR, "media"))]
        for i in file_names:
            if 'temp_' in i:
                os.remove(os.path.join(BASE_DIR, "media") + '/' + i)

        model_path = os.path.join(BASE_DIR, "media") + '/temp_' + str(self.cost_type) + '-' + str(method_name)
        model_name = str(self.cost_type) + '-' + str(method_name)
        joblib.dump(fit1, model_path)

        if ModelSave.objects.filter(model_name=model_name).exists():
            mdl = ModelSave.objects.filter(model_name=model_name)
            mdl.update(model_mape=mape_hw)

        else:
            mdl = ModelSave.objects.create(model_name=model_name, model_mape=mape_hw)
            mdl.save()

        return self.training_set.astype(float), self.testing_set.astype(float), y_hat, mape_hw


class LSTMInventory:

    def __init__(self, data, cost_type, time_start, time_end, train_start, train_end, predict_start, predcit_end):

        self.data = data.astype(float)
        self.orig_data = data.astype(float)
        self.time_interval = pd.date_range(start=time_start, end=time_end, freq='MS')  # generate the timestamp'
        self.data = self.data.set_index(self.time_interval)  # set the timestamp as a index

        self.orig_data = self.orig_data.set_index(self.time_interval)  # set the timestamp as a index
        self.data_timeseries = (self.data[cost_type])  # select the cost type from dataframe
        self.cost_type = cost_type
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.train_len = len(pd.date_range(start=train_start, end=train_end, freq='MS'))
        self.predict_start = pd.to_datetime(predict_start)
        self.predcit_end = pd.to_datetime(predcit_end)
        self.predcit_interval = pd.date_range(start=self.train_end, end=predcit_end, freq='MS')
        self.delay = 12
        self.batch_size = 10  # 要用多少批量來建模以及預測
        self.epoch = 2  # 要訓練幾回合
        self.length = 24  # 要用過去幾個月作為一個區間進行模型訓練與預測
        self.sampling_rate = 1  # 要用幾個月採樣一次
        self.stride = 1

    def data_preprocessing(self):
        self.mean_data = self.data.mean(axis=0)
        self.std_data = self.data.std(axis=0)
        self.data = (self.data - self.mean_data) / self.std_data

        target_tmp = self.data.values

        self.data = self.data.values

        print(self.data[0])

        training_data = list()
        target_t = list()

        for i in range(self.data.shape[0]):
            training_data.append([self.data[i][0]])
            target_t.append(target_tmp[i])

        target = np.zeros(shape=(len(target_t), self.delay))
        for i in range(1, self.delay + 1):
            temp_t = np.roll(target_t, -i, axis=0)
            for j in range(len(training_data)):
                target[j, i - 1] = temp_t[j]

        self.train_gen = TimeseriesGenerator(training_data, target,
                                             length=self.length,
                                             sampling_rate=1,
                                             stride=self.stride,
                                             start_index=0,
                                             end_index=self.train_len,
                                             batch_size=self.batch_size)

        self.testing_gen = TimeseriesGenerator(training_data[len(target_t) - self.delay - self.length:len(target_t)],
                                               target[len(target_t) - self.delay - self.length:len(target_t)],
                                               length=self.length,
                                               sampling_rate=1,
                                               stride=len(target_t) - (len(target_t) - self.delay - self.length) - 1,
                                               start_index=0,
                                               end_index=len(target_t) - (len(target_t) - self.delay - self.length) - 1,
                                               batch_size=self.batch_size)

    def lstm(self):
        method_name = 'LSTM'
        lstm_model = Sequential()
        lstm_model.add(layers.LSTM(32, return_sequences=True,
                                   input_shape=(self.length, 1)))  # ←由於處理完的資料是3維的, 所以第一層要用展平層, 將資料展平
        lstm_model.add(layers.LSTM(32, activation='relu'))
        lstm_model.add(layers.Dense(self.delay))
        lstm_model.summary()

        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        lstm_model.fit(self.train_gen, epochs=self.epoch, validation_data=self.testing_gen, verbose=1)


        y_hat = pd.Series(lstm_model.predict(self.testing_gen)[0],
                             index=pd.date_range(start=self.train_end+ relativedelta(months=1), end=self.train_end+ relativedelta(months=12), freq='MS'))


        y_hat = y_hat.values * self.std_data.values
        y_hat = y_hat + self.mean_data.values
        y_hat = pd.Series(y_hat,
                          index=pd.date_range(start=self.train_end + relativedelta(months=1),
                                              end=self.train_end + relativedelta(months=12), freq='MS'))


        self.training_set = self.orig_data[(self.orig_data.index <= self.train_end)]
        self.training_set = self.training_set.squeeze()


        self.testing_set = self.orig_data[(self.orig_data.index > self.train_end )]
        self.testing_set = self.testing_set[(self.testing_set.index <= self.train_end + relativedelta(months=12))]
        self.testing_set = self.testing_set.squeeze()


        mape_lstm = self.mape(np.array(self.testing_set.astype(float)), np.array(y_hat.astype(float)))



        file_names = [fn for fn in os.listdir(os.path.join(BASE_DIR, "media"))]
        for i in file_names:
            if 'temp_' in i:
                os.remove(os.path.join(BASE_DIR, "media") + '/' + i)

        model_path = os.path.join(BASE_DIR, "media") + '/temp_' + str(self.cost_type) + '-' + str(
            method_name) + '.h5'
        model_name = str(self.cost_type) + '-' + str(method_name)
        lstm_model.save(model_path)

        if ModelSave.objects.filter(model_name=model_name).exists():
            mdl = ModelSave.objects.filter(model_name=model_name)
            mdl.update(model_mape=mape_lstm)

        else:
            mdl = ModelSave.objects.create(model_name=model_name, model_mape=mape_lstm)
            mdl.save()

        for i in self.mean_data.index:
            if ModelStandardized.objects.filter(model_name=model_name, feature=i, flag=False).exists():
                mdl = ModelStandardized.objects.filter(model_name=model_name, feature=i, flag=False)
                mdl.update(mean=self.mean_data[i])
                mdl.update(std=self.std_data[i])

            else:
                mdl = ModelStandardized.objects.create(model_name=model_name, feature=i, mean=self.mean_data[i],
                                                       std=self.std_data[i], flag=False)
                mdl.save()

        self.training_set = self.training_set[(self.training_set.index >= self.train_start)]
        self.training_set = self.training_set[(self.training_set.index <= self.train_end)]

        return self.training_set.astype(float), self.testing_set.astype(float), y_hat, mape_lstm

    def mape(self, actual, pred):
        return np.mean(np.abs((actual - pred) / actual)) * 100


class ModelLoadPred:

    def __init__(self, data, cost_type, period_type, algorithm, predict_start, predcit_end):

        self.cost_type = cost_type
        self.algorithm = algorithm
        self.predict_start = predict_start
        self.predict_end = predcit_end
        self.predcit_interval = pd.date_range(start=predict_start, end=predcit_end, freq='MS')

        self.data = data
        self.period_type = period_type

        self.delay = 12
        self.batch_size = 10  # 要用多少批量來建模以及預測
        self.epoch = 2  # 要訓練幾回合
        self.length = 24  # 要用過去幾個月作為一個區間進行模型訓練與預測
        self.sampling_rate = 1  # 要用幾個月採樣一次
        self.stride = 1

    def load_pred(self):

        if self.period_type == '0':



            if self.algorithm == '0':
                method_name = 'SARIMA'
                model_path = os.path.join(BASE_DIR, "media") + '/' + str(self.cost_type) + '-' + str(method_name)

                fit1 = joblib.load(model_path)
                y_hat = pd.DataFrame(fit1.predict(start=self.predict_start, end=self.predict_end, dynamic=True).values,
                                     index=self.predcit_interval, columns=[method_name])

            elif self.algorithm == '1':
                method_name = 'Holt_winters'
                model_path = os.path.join(BASE_DIR, "media") + '/' + str(self.cost_type) + '-' + str(method_name)

                fit1 = joblib.load(model_path)
                y_hat = pd.DataFrame(fit1.forecast(len(self.predcit_interval)), index=self.predcit_interval,
                                     columns=[method_name])


            elif self.algorithm == '2':
                method_name = 'LSTM'

                relevant_path = os.path.join(BASE_DIR, "media")

                # included_extensions = ['h5']
                # file_names = [fn for fn in os.listdir(relevant_path)
                #               if any(fn.endswith(ext) for ext in included_extensions)]
                # file_name = ''
                # for i in file_names:
                #     if self.cost_type == i.replace('.h5',''):
                #         file_name = i

                model_path = os.path.join(BASE_DIR, "media") + '/' + self.cost_type + str('-LSTM.h5')
                # print(model_path)

                fit1 = load_model(model_path)

                # model_name = file_name.replace('.h5','')

                md1 = ModelStandardized.objects.filter(model_name=self.cost_type + str('-LSTM'), feature=self.cost_type, flag=True)
                print('mean' + str(md1[0].mean))
                print('std' + str(md1[0].std))

                self.mean_data = float(md1[0].mean)
                self.std_data = float(md1[0].std)
                self.data = (self.data - self.mean_data) / self.std_data

                target_tmp = self.data.values

                self.data = self.data.values

                training_data = list()
                target_t = list()

                for i in range(self.data.shape[0]):
                    training_data.append([self.data[0]])
                    target_t.append(target_tmp[i])

                target = np.zeros(shape=(len(target_t), self.delay))
                for i in range(1, self.delay + 1):
                    temp_t = np.roll(target_t, -i, axis=0)
                    for j in range(len(training_data)):
                        target[j, i - 1] = temp_t[j]


                self.testing_gen = TimeseriesGenerator(
                    training_data[len(target_t) - self.delay - self.length:len(target_t)],
                    target[len(target_t) - self.delay - self.length:len(target_t)],
                    length=self.length,
                    sampling_rate=1,
                    stride=len(target_t) - (len(target_t) - self.delay - self.length) - 1,
                    start_index=0,
                    end_index=len(target_t) - (len(target_t) - self.delay - self.length) - 1,
                    batch_size=self.batch_size)

                time_start = datetime.strptime(self.predict_start, "%Y-%m-%d")

                time_end = time_start + relativedelta(months=11)
                time_interval = pd.date_range(start=time_start, end=time_end, freq='MS')  # generate the timestamp'

                y_hat = pd.DataFrame(fit1.predict(self.testing_gen)[0], index=time_interval, columns=[method_name])

                y_hat = y_hat * self.std_data
                y_hat = y_hat + self.mean_data

        elif self.period_type == '1':
            if self.algorithm == '0':
                method_name = 'RandomForest'

                model_path = os.path.join(BASE_DIR, "media") + '/' + str(self.cost_type) + '-' + str(
                    method_name) + '.mod'

                fit1 = joblib.load(model_path)
                y_hat =fit1.predict(self.data)[0]

            elif self.algorithm == '1':
                method_name = 'MARS'

                model_path = os.path.join(BASE_DIR, "media") + '/' + str(self.cost_type) + '-' + str(
                    method_name) + '.mod'

                model_name = str(self.cost_type) + '-' + str(method_name)

                md1 = ModelStandardized.objects.filter(model_name=model_name, flag=True)

                c = 0
                for i in md1:
                    if i.feature is not 'y':
                        self.data[0][c] = (self.data[0][c] - i.mean) / i.std
                        c = c + 1
                    else:
                        y_mean = float(i.mean)
                        y_std = float(i.std)

                fit1 = joblib.load(model_path)
                y_hat = fit1.predict(self.data)[0] * y_std + y_mean



        return y_hat


class ModelSaving:

    def __init__(self, cost_type, algorithm, period_type):
        self.cost_type = cost_type
        self.algorithm = algorithm
        self.period_type = period_type

    def modelSave(self):

        model_path = os.path.join(BASE_DIR, "media") + '/'
        file_names = [fn for fn in os.listdir(os.path.join(BASE_DIR, "media"))]
        file_name = ''

        if self.period_type == '0':
            if self.algorithm == '0':
                method_name = 'SARIMA'
            elif self.algorithm == '1':
                method_name = 'Holt_winters'
            elif self.algorithm == '2':
                method_name = 'LSTM'
                for i in file_names:
                    # if ('temp_' not in i) and (method_name in i) and (self.cost_type == i.replace('.h5')):
                    #     os.remove(model_path+i)
                    if ('temp_' in i) and (method_name in i):
                        model_name = i.replace('.h5','').replace('temp_','')

                        if ModelStandardized.objects.filter(model_name=model_name, flag=True).exists():
                            mdl = ModelStandardized.objects.filter(model_name=model_name, flag=True)
                            mdl.delete()

                        mdl = ModelStandardized.objects.filter(model_name=model_name)

                        mdl.update(flag=True)


        elif self.period_type == '1':
            if self.algorithm == '0':
                method_name = 'RandomForest'
            elif self.algorithm == '1':
                method_name = 'MARS'
                for i in file_names:
                    if ('temp_' not in i) and (method_name in i) and (self.cost_type in i):
                        os.remove(model_path+i)
                    elif ('temp_' in i) and (method_name in i):
                        model_name = i.replace('.mod','').replace('temp_','')
                        if ModelStandardized.objects.filter(model_name=model_name, flag=True).exists():
                            mdl = ModelStandardized.objects.filter(model_name=model_name, flag=True)
                            mdl.delete()

                        mdl = ModelStandardized.objects.filter(model_name=model_name)

                        mdl.update(flag=True)




        for i in file_names:
            if ('temp_' in i) and (method_name in i):
                file_name = i

        file_rename = file_name.replace('temp_', '')
        move(model_path + file_name, model_path + file_rename)


class MachineLearning:

    def __init__(self, X, y,cost_type, featurs):
        self.X = X.astype(float)
        self.y = y.astype(float)
        self.cost_type = cost_type
        self.featurs = featurs

    def randomForest(self):
        method_name = 'RandomForest'
        param_grid = {"n_estimators": range(100, 1001, 200),
                      'max_depth': range(2, 11, 2)}

        pipe = Pipeline(steps=[('rfr', GridSearchCV(RandomForestRegressor(), param_grid,cv=5, n_jobs=-1)),
                               ])



        pipe.fit(self.X.values, self.y.values.ravel())


        y_hat = pd.Series(pipe.predict(self.X.values), index=self.y.index)

        mape_rf = self.mape(np.array(self.y.astype(float)), np.array(y_hat.astype(float)))

        mape_rf = round(mape_rf, 2)

        file_names = [fn for fn in os.listdir(os.path.join(BASE_DIR, "media"))]
        for i in file_names:
            if 'temp_' in i:
                os.remove(os.path.join(BASE_DIR, "media") + '/' + i)

        model_path = os.path.join(BASE_DIR, "media") + '/temp_' + str(self.cost_type) + '-' + str(method_name) +'.mod'
        model_name = str(self.cost_type) + '-' + str(method_name)
        joblib.dump(pipe, model_path)

        if ModelSave.objects.filter(model_name=model_name).exists():
            mdl = ModelSave.objects.filter(model_name=model_name)
            mdl.update(model_mape=mape_rf)

        else:
            mdl = ModelSave.objects.create(model_name=model_name, model_mape=mape_rf)
            mdl.save()

        return self.y, y_hat, mape_rf

    # def MARS(self):

    #     method_name = 'MARS'

    #     x_mean = self.X.mean(axis = 0)
    #     x_std = self.X.std(axis = 0)
    #     self.X = (self.X-x_mean)/x_std

    #     y_mean = self.y.mean()
    #     y_std = self.y.std()
    #     self.y = (self.y - y_mean) / y_std



    #     pipe = Pipeline(steps=[("MARS_model", pyearth.Earth())])

    #     pipe.fit(self.X, self.y)


    #     y_hat = pd.Series(pipe.predict(self.X), index=self.y.index)

    #     self.y = (self.y * y_std) + y_mean
    #     y_hat = (y_hat * y_std) + y_mean




    #     mape_mars = self.mape(np.array(self.y.astype(float)), np.array(y_hat.astype(float)))



    #     file_names = [fn for fn in os.listdir(os.path.join(BASE_DIR, "media"))]
    #     for i in file_names:
    #         if 'temp_' in i:
    #             os.remove(os.path.join(BASE_DIR, "media") + '/' + i)

    #     model_path = os.path.join(BASE_DIR, "media") + '/temp_' + str(self.cost_type) + '-' + str(
    #         method_name) + '.mod'
    #     model_name = str(self.cost_type) + '-' + str(method_name)

    #     joblib.dump(pipe, model_path)

    #     if ModelSave.objects.filter(model_name=model_name).exists():
    #         mdl = ModelSave.objects.filter(model_name=model_name)
    #         mdl.update(model_mape=mape_mars)

    #     else:
    #         mdl = ModelSave.objects.create(model_name=model_name, model_mape=mape_mars)
    #         mdl.save()



    #     for i in x_mean.index:
    #         if ModelStandardized.objects.filter(model_name=model_name, feature=i, flag=False).exists():
    #             mdl = ModelStandardized.objects.filter(model_name=model_name, feature=i, flag=False)
    #             mdl.update(mean=x_mean[i])
    #             mdl.update(std=x_std[i])

    #         else:
    #             mdl = ModelStandardized.objects.create(model_name=model_name, feature=i, mean=x_mean[i],
    #                                                    std=x_std[i])
    #             mdl.save()

    #     if ModelStandardized.objects.filter(model_name=model_name, feature='y', flag=False).exists():
    #         mdl = ModelStandardized.objects.filter(model_name=model_name, feature='y')
    #         mdl.update(mean=y_mean)
    #         mdl.update(std=y_std)
    #     else:
    #         mdl = ModelStandardized.objects.create(model_name=model_name, feature='y', mean=y_mean, std=y_std, flag=False)

    #         mdl.save()

    #     return self.y, y_hat, mape_mars


    def mape(self, actual, pred):
        return np.mean(np.abs((actual - pred) / actual)) * 100
