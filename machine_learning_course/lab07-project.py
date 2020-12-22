import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def read_temp_mid_sn() -> int:
    with open('data/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]
    return sn_temp_mid


def load_dataframe(path, value) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df.rename(columns={'value': value}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.drop(columns=['unit'], inplace=True)
    df.set_index('time', inplace=True)
    return df


def main():
    # Load DataFrames
    df_temp = load_dataframe('data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv', 'temp')
    sn_temp_mid = read_temp_mid_sn()
    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]
    df_target_temp = load_dataframe('data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv',
                                    'target_temp')
    df_valve = load_dataframe('data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv', 'valve')

    # Resample all Data
    df_combined = pd.concat([df_temp, df_target_temp, df_valve])
    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    # Ground truth
    df_combined['temp_gt'] = df_combined['temp']

    # Create train & test DataFrame
    mask_train = (df_combined.index < '2020-10-27')
    mask_test = (df_combined.index > '2020-10-27') & (df_combined.index <= '2020-10-28')
    df_combined['temp_last'] = df_combined['temp'].shift(1, fill_value=21)
    df_train = df_combined.loc[mask_train]
    df_test = df_combined.loc[mask_test]

    # Last 15 min predition
    y_last = df_test['temp_last'].to_numpy()[1:-1]

    # RandomForest predition
    X_train = df_train[['temp', 'valve']].to_numpy()[1:-1]
    y_train = df_train['temp_gt'].to_numpy()[1:-1]
    X_test = df_test[['temp', 'valve']].to_numpy()
    reg_rf = RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)
    y_predicted = reg_rf.predict(X_test)

    # Score of preditions
    y_test = df_test['temp_gt'].to_numpy()[1:-1]
    print(f'last MAE: ', "{:5.5f}".format(metrics.mean_absolute_error(y_test, y_last)))
    print(f'last MSE: ', "{:5.5f}".format(metrics.mean_absolute_error(y_test, y_predicted[1:-1])))
    print(f'pred MAE: ', "{:5.5f}".format(metrics.mean_squared_error(y_test, y_last)))
    print(f'pred MSE: ', "{:5.5f}".format(metrics.mean_squared_error(y_test, y_predicted[1:-1])))

    # Visualisation
    df_visual = df_test.copy()
    df_visual['temp_predicted'] = y_predicted.tolist()
    df_visual.drop(columns=['valve', 'temp', 'target_temp'], inplace=True)
    df_visual.plot()
    plt.show()


if __name__ == '__main__':
    main()
