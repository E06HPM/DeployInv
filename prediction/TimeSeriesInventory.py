import numpy as np  
from scipy import stats  
import pandas as pd  
import matplotlib.pyplot as plt  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt, SARIMAX, ARIMA
import warnings
import platform

warnings.filterwarnings("ignore")

# if platform.system()=='Darwin':
#
#     plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常顯示中文
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#     plt.rcParams['font.serif'] = ['Arial Unicode MS']
#     plt.rcParams['axes.unicode_minus'] = False
# else:
#     plt.rcParams['font.sans-serif']=['Microsoft YaHei']
#     plt.rcParams['axes.unicode_minus'] = False
#
#


class TimeSeriesInventory:


    def __init__(self, data,cost_type ,time_start,time_end,train_start,train_end,predict_start,predcit_end):

        self.data = data
        print(data)
        self.time_interval = pd.date_range(start = time_start, end = time_end, freq='MS') #generate the timestamp'
        self.data = self.data.set_index(self.time_interval) #set the timestamp as a index
        self.data_timeseries = (self.data[cost_type])#select the cost type from dataframe
        self.cost_type = cost_type
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.predict_start = pd.to_datetime(predict_start)
        self.predcit_end = pd.to_datetime(predcit_end)
        self.predcit_interval = pd.date_range(start = predict_start, end = predcit_end, freq='MS')


    # Split the original data into training & testing set
    def data_split(self):
        self.training_set = self.data_timeseries[(self.data_timeseries.index <= self.train_end)]
        self.testing_set = self.data_timeseries[(self.data_timeseries.index > self.train_end)]


    # Cacluate auto-correlation and partial auto-correlation
    def acf_pacf(self):
        ax1 =plot_acf(self.training_set, lags = 50)
        ax2=plot_pacf(self.training_set, lags = 50)

    # Plot time series plot
    def plot_data(self):
        ax = plt.figure(figsize=(7,5))
        ax = self.training_set.plot(label = 'Training', fontsize=14)
        if not self.testing_set.empty:
            ax = self.testing_set.plot(label = 'Testing', fontsize=14)
        ax.legend(loc='best')
        ax.set_xlabel('時間')
        ax.set_ylabel(self.cost_type)
        plt.show()

    # Caculas mean average percentage error
    def mape(self,actual, pred):
        return np.mean(np.abs((actual - pred) / actual)) * 100

    # Show the training and testing result
    def result_plot(self, method_name, y_hat):
        plt.figure(figsize=(7, 5))
        plt.plot(self.training_set.index, self.training_set, label='Training')
        if not self.testing_set.empty:
            plt.plot(self.testing_set.index, self.testing_set, label='Testing')
        plt.plot(y_hat.index, y_hat[method_name], label=method_name)
        plt.legend(loc='best')
        plt.title(method_name+'預測結果')
        plt.xlabel('時間')
        plt.ylabel(self.cost_type)
        plt.show()
        actual, pred = self.testing_set.copy(), y_hat[method_name]
        if not self.testing_set.empty:
            print("MAPE({0})={1:.2f}%".format(method_name, self.mape(actual, pred)))
        print("預測值")
        print(y_hat[method_name])

    # Naive Method
    def naive(self):
        method_name = 'Naive'
        train = np.asarray(self.training_set)
        y_hat = pd.DataFrame(train[len(train) - 1], index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)

    # Simple-average Method
    def simple_average(self):
        method_name = 'Average'
        y_hat = pd.DataFrame(self.training_set.mean(), index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)

    # Moving average method
    def moving_average(self, periods_base = 12):
        method_name = 'Moving Average'
        y_hat = pd.DataFrame(self.training_set.rolling(periods_base).mean().iloc[-1], index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)

    #Simple Expsmoothing method
    def simple_expsmoothing(self):
        method_name = 'Simple Expo Smoothing'
        fit = SimpleExpSmoothing(np.asarray(self.training_set)).fit(smoothing_level=0.6, optimized=False)
        y_hat = pd.DataFrame(fit.forecast(len(self.predcit_interval)),index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)

    # Holt linear Method
    def holt(self):
        method_name = 'Holt_linear'
        fit = Holt(np.asarray(self.training_set)).fit(smoothing_level=0.3, smoothing_slope=0.1)
        y_hat = pd.DataFrame( fit.forecast(len(self.predcit_interval)), index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)

    # Holt Winters method
    def holt_winters(self):
        method_name = 'Holt linear'
        fit1 = ExponentialSmoothing(np.asarray(self.training_set), seasonal_periods=12, trend='add', seasonal='add', ).fit()
        y_hat = pd.DataFrame( fit1.forecast(len(self.predcit_interval)), index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)

    # SARIMA method
    def sarima(self):
        method_name = 'SARIMA'
        fit1 = SARIMAX(self.training_set, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
        y_hat = pd.DataFrame(fit1.predict(start=self.predict_start, end=self.predcit_end, dynamic=True).values, index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)
        return y_hat

    # ARIMA method
    def arima(self):
        method_name = 'ARIMA'
        fit1 = ARIMA(self.training_set, order=(3, 2, 1)).fit(disp=-1)
        fc, se, conf = fit1.forecast(len(self.predcit_interval), alpha=0.05)
        y_hat = pd.DataFrame(fc, index = self.predcit_interval, columns=[method_name])
        self.result_plot(method_name, y_hat)

    def analyze(self):
        self.data_split()
        self.acf_pacf()
        self.plot_data()


# data_path = r'/Users/dannyhuang/Documents/博士班/專案/漢翔/存貨預測/時間序列輸入格式.xlsx'

# time_start = '2010-1-1'         #輸入資料起始時間
# time_end = '2020-4-1'           #輸入資料結束時間
# train_start = '2010-1-1'        #決定訓練集要起始時間
# train_end = '2020-4-1'          #決定訓練集要結束時間
# predict_start = '2020-5-1'      #決定測試集起始時間，也就是要預測區間的起始時間
# predcit_end = '2021-4-1'        #決定測試集結束時間，也就是要預測區間的結束時間
# cost_type = '在製存貨金額'          #決定要預測哪一類的存貨金額：原物料存貨金額(不含WRITE-OFF)、半製品存貨金額(不含WRITE-OFF)、在製品存貨金額(不含WRITE-OFF)、總存貨金額(不含WRITE-OFF)、原物料存貨金額、半製存貨金額、在製存貨金額、總存貨金額
# #建立result物件，用以分析存貨成本的資料模式屬於序列型態，以決定採用的模型（函數名稱）
# result_new = time_series(data_path,cost_type,time_start,time_end,train_start,train_end,predict_start,predcit_end)
# #依據下圖結果，對應到圖一，可判斷此筆成本屬於趨勢型，故可採用arima、sarima、holt、holt_winters等模型
# result_new.analyze()

# result_new.naive()
