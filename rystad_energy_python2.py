#Libraries Import
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import requests
import datetime
import io
import matplotlib.pyplot as plt
from matplotlib import rcParams


# personal key to access the api
key = "4915003e3c8e7a7db077368f7f7a4792"

# essential urls to fetch the unknown data
net_product_import_url = "http://api.eia.gov/series/?api_key="+ key +"&series_id=PET.MTPNTUS2.M"
storage_draw_url = "http://api.eia.gov/series/?api_key="+ key +"&series_id=PET.MCRSTUS1.M"
refinery_gains_url = "http://api.eia.gov/series/?api_key="+key+"&series_id=PET.MPGRYUS3.M"

#function to create df out of data received from the website
def get_time_series_df(url):
  response = requests.get(url)
  response_data = response.json().get('series')[0]
  df_name = response_data.get('name')
  data = response_data.get('data')
  time_period = []
  quantity = []
  for i in data:
    time_period.append(i[0])
    quantity.append(i[1])
  df = pd.DataFrame({
      "Date": time_period,
      "Value": quantity
  })
  df['Year'] = df.Date.apply(lambda x: int(x[:4]))
  df['Month'] = df.Date.apply(lambda x: x[4:])
  df['Date'] = df.apply(lambda x: str(x['Year']) + "-" + str(x['Month']), axis=1)
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.sort_values("Date")
  return df


# this function applies the LSTM model on the DataFrame passed as parameter and returns forecasted dataframe with the passed dataframe
def apply_lstm(df):
  #Separate dates for future plotting
  train_dates = pd.to_datetime(df['Date'])
  #Variables for training
  cols = list(df)[1:2]
  #New dataframe with only training data - 1 column
  df_for_training = df[cols].astype(float)
  #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
  # normalize the dataset  
  scaler = StandardScaler()
  scaler = scaler.fit(df_for_training)
  df_for_training_scaled = scaler.transform(df_for_training)

  #Empty lists to be populated using formatted training data
  trainX = []
  trainY = []

  # Number of days we want to look into the future based on the past days.
  n_future = 1
  # Number of past days we want to use to predict the future.
  n_past = 8

  #Reformat input data into a shape: (n_samples x timesteps x n_features)
  for i in range(n_past, len(df_for_training_scaled) - n_future+1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future-1:i + n_future, 0])
  trainX, trainY = np.array(trainX), np.array(trainY)

  # define the model
  model = Sequential()
  model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
  model.add(LSTM(32, activation='relu', return_sequences=False))
  model.add(Dropout(0.2))
  model.add(Dense(trainY.shape[1]))
  model.compile(optimizer='adam', loss='mse')
  model.summary()
  # fit the model
  history = model.fit(trainX, trainY, epochs=10, batch_size=9, validation_split=0.1, verbose=1)
  n_future=54
  forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1m').tolist()
  #Make prediction
  forecast = model.predict(trainX[-n_future:])
  #Perform inverse transformation to rescale back to original range
  forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=1)
  y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]
  forecast_dates = []
  # Convert timestamp to date
  for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
  df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Value': y_pred_future})
  df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
  return df_forecast, df


# get lstm applied dataframe and further append them with the given data
net_products_import_predictions, net_products_import = apply_lstm(get_time_series_df(net_product_import_url))  
storage_draw_predictions, storage_draw = apply_lstm(get_time_series_df(storage_draw_url))  
refinery_gains_predictions, refinery_gains= apply_lstm(get_time_series_df(refinery_gains_url))  

def join_df(prediction, df):
  df = df[['Date', 'Value']]
  main_df = df.append(prediction[1:], ignore_index=True)
  main_df['Year'] = main_df.Date.apply(lambda x: x.year)
  main_df['Month'] = main_df.Date.apply(lambda x: datetime.datetime.strptime(str(x.month), "%m").strftime("%B"))
  main_df = main_df[main_df['Year']>=2013][['Year', 'Month', 'Value']]
  return main_df

#joining the predicted data with given and further merging all the data into one dataframe
final_net_import_df = join_df(net_products_import_predictions, net_products_import)
final_net_import_df = final_net_import_df.rename(columns={'Value': 'Net_products_input'})
final_storage_draw_df = join_df(storage_draw_predictions, storage_draw)
final_storage_draw_df = final_storage_draw_df.rename(columns={'Value': 'Net_storage_draw'})
final_refinery_gains = join_df(refinery_gains_predictions, refinery_gains)
final_refinery_gains = final_refinery_gains.rename(columns={'Value': 'refinery_gains'})
final_df = final_net_import_df.merge(final_storage_draw_df, on=['Year', 'Month'], how='left')
final_df = final_df.merge(final_refinery_gains, on=['Year', 'Month'], how='left')

df2 = pd.read_excel("Rystad Energy Products demand.xlsx")
df3 = df2.merge(final_df, on=['Year', 'Month'], how='left')

#code copied from previous assignment submitted
def get_dataset(padd_no):
    # personal api key
    api_key = "4915003e3c8e7a7db077368f7f7a4792"
    url = "http://api.eia.gov/series/?api_key=" + api_key + "&series_id=PET.MCRRIP" + str(padd_no) + "2.M"
    response = requests.get(url)
    response_data = response.json().get('series')[0]
    df_name = response_data.get('name')
    frequency = 'M'
    units = 'Thousand Barrels per Day'
    data = response_data.get('data')
    time_period = []
    quantity = []
    for i in data:
        time_period.append(i[0])
        quantity.append(i[1])
    df = pd.DataFrame({
        'Period': time_period,
        'Value': quantity,
    })
    df['Series_Name'] = df_name
    df['Frequency'] = frequency
    df['Units'] = units
    df['Year'] = df.Period.apply(lambda x: int(x[:4]))
    df['Month'] = df.Period.apply(lambda x: int(x[4:]))
    df['quarter'] = df.apply(lambda x: x['Month'] // 3 if x['Month'] % 3 == 0 else x['Month'] // 3 + 1, axis=1)
    # df.to_csv("padd1_data.csv", index=False)
    return df


def data_processing():
    # fetching dataframe from link part(a)
    padd1_df = get_dataset(1)
    padd2_df = get_dataset(2)
    padd3_df = get_dataset(3)
    padd4_df = get_dataset(4)
    padd5_df = get_dataset(5)

    # taking only entries greater than or equal to 2013 part(b)
    padd1_df_2013_onwards = padd1_df[padd1_df['Year'] >= 2013][['Year', 'Month', 'quarter', 'Value']]
    padd2_df_2013_onwards = padd2_df[padd2_df['Year'] >= 2013][['Year', 'Month', 'quarter', 'Value']]
    padd3_df_2013_onwards = padd3_df[padd3_df['Year'] >= 2013][['Year', 'Month', 'quarter', 'Value']]
    padd4_df_2013_onwards = padd4_df[padd4_df['Year'] >= 2013][['Year', 'Month', 'quarter', 'Value']]
    padd5_df_2013_onwards = padd5_df[padd5_df['Year'] >= 2013][['Year', 'Month', 'quarter', 'Value']]

    # merging data to get all padd's data in one dataframe part(c)
    padd1_padd2_merged_df = padd1_df_2013_onwards.merge(padd2_df_2013_onwards, on=['Year', 'quarter', 'Month'],
                                                        how='left',
                                                        suffixes=('_padd1', '_padd2'))
    padd3_padd4_merged_df = padd3_df_2013_onwards.merge(padd4_df_2013_onwards, on=['Year', 'quarter', 'Month'],
                                                        how='left',
                                                        suffixes=('_padd3', '_padd4'))
    padd3_padd4_padd5_merged_df = padd3_padd4_merged_df.merge(padd5_df_2013_onwards, on=['Year', 'quarter', 'Month'],
                                                              how='left')
    merged_df = padd1_padd2_merged_df.merge(padd3_padd4_padd5_merged_df, on=['Year', 'quarter', 'Month'], how='left')

    # renaming the fifth padd column to 'Value_padd5'
    merged_df = merged_df.rename(columns={"Value": "Value_padd5"})

    # suming up total crude oil input of all padd's part(d)
    merged_df['total_us_refinery_net_input'] = merged_df.apply(
        lambda x: x['Value_padd1'] + x['Value_padd2'] + x['Value_padd3'] + x['Value_padd4'] + x['Value_padd5'], axis=1)
    total_us_refinery_net_input_of_crude_oil_df = merged_df[['Year', 'quarter', 'Month', 'total_us_refinery_net_input']]

    # summarizing monthly data by 'quarter' part(e)
    total_us_refinery_net_input_of_crude_oil_quarterly_df = merged_df.groupby(['Year', 'quarter']).agg({
        "Value_padd1": sum,
        "Value_padd2": sum,
        "Value_padd3": sum,
        "Value_padd4": sum,
        "Value_padd5": sum,
        "total_us_refinery_net_input": sum,
    }).reset_index()

    # summarizing monthly data by 'year' part(f)
    total_us_refinery_net_input_of_crude_oil_yearly_df = merged_df.groupby(['Year']).agg({
        "Value_padd1": sum,
        "Value_padd2": sum,
        "Value_padd3": sum,
        "Value_padd4": sum,
        "Value_padd5": sum,
        "total_us_refinery_net_input": sum,
    }).reset_index()

    total_us_refinery_net_input_of_crude_oil_monthly_df = merged_df.groupby(['Year', 'quarter', 'Month']).agg({
        "Value_padd1": sum,
        "Value_padd2": sum,
        "Value_padd3": sum,
        "Value_padd4": sum,
        "Value_padd5": sum,
        "total_us_refinery_net_input": sum,
    }).reset_index()
    return total_us_refinery_net_input_of_crude_oil_monthly_df


r = data_processing()
r['Month'] =  r.Month.apply(lambda x: datetime.datetime.strptime(str(x), "%m").strftime("%B"))
r = r[['Year', 'Month', 'total_us_refinery_net_input']]
df4 = df3.merge(r, on=['Year', 'Month'], how='left')
df4['Net_storage_draw_diff'] = 0

df4['Net_storage_draw_diff'] = 0
for i in range(len(df4)):
  if i ==0 or i == len(df4) - 1:
    df4.at[i, ['Net_storage_draw_diff']] = 0    
  else:
  	#dividing the data with 30, to convert it to thousand barrels per day from thousand barrels per month
    df4.at[i, ['Net_storage_draw_diff']] = (df4.at[i+1, 'Net_storage_draw'] - df4.at[i, 'Net_storage_draw'])/30

#function calculates the refinery input on the basis of equation given in case_study
def calcc_refinery_input(row):
  if row['Net_products_input'] > 0:
    net_product_input_const = row['Net_products_input']
  else:
    net_product_input_const = 0

  if row['Net_storage_draw_diff'] < 0:
    net_storage_draw_diff = row['Net_storage_draw_diff']
  else:
    net_storage_draw_diff = 0
  x = row['End user total products demand '] - 0.8*(row['LPG']) - row['Direct crude burn'] - row['Biofuels'] - net_product_input_const + net_storage_draw_diff
  refinery_input = x/(1-(row['refinery_gains']/100))
  return refinery_input

df4['net_input_refinery_predicted_using_equation'] = df4.apply(lambda x: calcc_refinery_input(x), axis=1)
df5 = df4[df4.total_us_refinery_net_input.notna()]
mse = math.sqrt(np.mean((df5['total_us_refinery_net_input'] - df5['net_input_refinery_predicted_using_equation'])**2))

df4['month_num'] = df4.Month.apply(lambda x: datetime.datetime.strptime(x, "%B").month)
from datetime import date
df4['Date'] = df4.apply(lambda x: date(x['Year'], x['month_num'], 1), axis=1)
df6 = df4[['Date', 'net_input_refinery_predicted_using_equation', 'total_us_refinery_net_input']]
df6 = df6.set_index('Date')

rcParams['figure.figsize'] = 25, 8
plt.plot(df6.total_us_refinery_net_input, label='given_net_input')
plt.plot(df6.net_input_refinery_predicted_using_equation, label='prediction')
plt.legend(loc=2)
plt.xlabel("Date")
plt.ylabel("Thousand Barrels per day")
plt.title("Chart bw given net input and predicted input to refinery")
plt.grid(True)
plt.savefig('Rystad_energy_refinery_assignment_plot.png')


df4