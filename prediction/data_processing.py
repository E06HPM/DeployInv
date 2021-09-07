import json
import pandas as pd


def data_for_ploting(data):

    output = dict()
    output['year'] = list()
    output['month'] = list()
    output['day'] = list()
    output['value'] = list()

    for index, value in data.items():

        output['year'].append(index.date().year)
        output['month'].append(index.date().month)
        output['day'].append(index.date().day)
        output['value'].append(value)

    output = json.dumps(output)

    return output

def history_data_covert(data,cost_type):

    data['date'] = pd.to_datetime(data['date'])
    data[cost_type] = data[cost_type].astype(float)
    data = data.set_index('date')
    data = data.squeeze()

    return data