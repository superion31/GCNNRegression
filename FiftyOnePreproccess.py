#importing libraries
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from scipy.stats import iqr
import re
import os

#function for general categorization


def extrafeatures33(row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2):
  if row['phase'] == 'climbing_BELOW-FL100':
    return 'Drop1'
  elif row['phase'] == 'climbing_ABOVE-FL100' and row['h[ft]'] < cli_lim1:
    return 'Climbing(-)'
  elif row['phase'] == 'climbing_ABOVE-FL100' and row['h[ft]'] >= cli_lim1 and row['h[ft]'] < cli_lim2:
    return 'Climbing'
  elif row['phase'] == 'climbing_ABOVE-FL100' and row['h[ft]'] >= cli_lim2:
    return 'Climbing(+)'
  elif row['phase'] == 'descending_ABOVE-FL100' and row['h[ft]'] < desc_lim1:
    return 'Descending(-)'
  elif row['phase'] == 'descending_ABOVE-FL100' and row['h[ft]'] >= desc_lim1 and row['h[ft]'] < desc_lim2:
    return 'Descending'
  elif row['phase'] == 'descending_ABOVE-FL100' and row['h[ft]'] >= desc_lim2:
    return 'Descending(+)'
  elif row['phase'] == 'descending_BELOW-FL100':
    return 'Drop2'
  elif row['phase'] == 'cruising' and row['h[ft]'] < cru_lim1:
    return 'Crusing(-)'
  elif row['phase'] == 'cruising' and row['h[ft]'] >= cru_lim1 and row['h[ft]'] < cru_lim2:
    return 'Cruising'
  elif row['phase'] == 'cruising' and row['h[ft]'] > cru_lim2:
    return 'Cruising(+)'

#calculating distance using longitude and latidute


def getDist(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * \
        cos(rLat2) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c  # Distance in km
    return d

#calculating valocity


def getVel(dist_km, time_start, time_end):
    """Return 0 if time_start == time_end, avoid dividing by 0"""
    return dist_km / (time_end - time_start) if time_end > time_start else 0

#adding distances column


def Dist_col(df):
  distances = []

  for i in range(len(df)):
    if i > 0:
      distances.append(
          1000 * getDist(df['latitude'][i-1], df['longitude'][i-1], df['latitude'][i], df['longitude'][i]))
    else:
      distances.append(0)
  df['Distance[m]'] = distances

  output_df = df.drop([0])
  output_df = output_df.reset_index(drop=True)

  return output_df

#adding velocities column


def Vel_col(df):
  velocities = []

  for i in range(len(df)):
    if i > 0:
      velocities.append(
          getVel(df['Distance[m]'][i], df['posix'][i-1], df['posix'][i]))
    else:
      velocities.append(0)

  df['Velocities[m/s]'] = velocities

  output_df = df.drop([0])
  output_df = output_df.reset_index(drop=True)

  return output_df

#adding velocities and geometric altitude derivatives columns


def vhdots_col(df):
  vdots = []
  hdots = []

  for i in range(len(df)):
    if i > 0:
      vdots.append(1.9438444924423 * (df['Velocities[m/s]'][i]
                   - df['Velocities[m/s]'][i-1]) / (df['posix'][i] - df['posix'][i-1]))
      hdots.append(60 * (df['h[ft]'][i] - df['h[ft]'][i-1]
                         ) / (df['posix'][i] - df['posix'][i-1]))
    else:
      vdots.append(0)
      hdots.append(0)

  df['vdot[kt/s]'] = vdots
  df['hdot[ft/min]'] = hdots

  output_df = df.drop([0])
  output_df = output_df.reset_index(drop=True)

  return output_df

#cleaning columns including infinities


def drop_inf(df):
  lst = []
  for i in range(len(df)):
    if df['vdot[kt/s]'][i] == float('inf') or df['vdot[kt/s]'][i] == float('-inf'):
      lst.append(i)

  df.drop(lst, axis=0, inplace=True)
  df.reset_index(drop=True)
  return df

#final concatenated dataframe


def data_prep(name):

  pre_df = pd.read_csv(name)
  pre_df = Dist_col(pre_df)
  pre_df = Vel_col(pre_df)
  pre_df = vhdots_col(pre_df)
  pre_df = drop_inf(pre_df)

  time_keep = ['phase', 'posix', 'h[ft]']
  time_df = pre_df[time_keep]
  duration = []

  to_keep = ['phase', 'h[ft]', 'AirTemperature(C)', 'v_wind_component',
             'u_wind_component', 'latitude', 'longitude', 'vdot[kt/s]', 'hdot[ft/min]']

  df = pre_df[to_keep]
  df.columns = ['phase', 'h[ft]', 'Temp[oC]', 'v-Wn[kt]',
                'u-We[kt]', 'Lat[o]', 'Lon[o]', 'vdot[kt/s]', 'hdot[ft/min]']

  delta_cli = (df[df['phase'] == 'climbing_ABOVE-FL100']['h[ft]'].max()
               - df[df['phase'] == 'climbing_ABOVE-FL100']['h[ft]'].min())/3
  cli_lim1 = df[df['phase'] == 'climbing_ABOVE-FL100']['h[ft]'].min() + \
      delta_cli
  cli_lim2 = df[df['phase'] == 'climbing_ABOVE-FL100']['h[ft]'].min() + \
      2*delta_cli

  delta_cru = (df[df['phase'] == 'cruising']['h[ft]'].max()
               - df[df['phase'] == 'cruising']['h[ft]'].min())/3
  cru_lim1 = df[df['phase'] == 'cruising']['h[ft]'].min() + delta_cru
  cru_lim2 = df[df['phase'] == 'cruising']['h[ft]'].min() + 2 * delta_cru

  delta_desc = (df[df['phase'] == 'descending_ABOVE-FL100']['h[ft]'].max()
                - df[df['phase'] == 'descending_ABOVE-FL100']['h[ft]'].min())/3
  desc_lim1 = df[df['phase']
                 == 'descending_ABOVE-FL100']['h[ft]'].min() + delta_desc
  desc_lim2 = df[df['phase']
                 == 'descending_ABOVE-FL100']['h[ft]'].min() + 2 * delta_desc

  df_median = df.copy()
  df_iqr = df.copy()

  time_df['extrafeatures'] = time_df.apply(lambda row: extrafeatures33(
      row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2), axis=1)
  time_df = time_df.groupby('extrafeatures').agg(['min', 'max'])
  for i in range(len(time_df)):
    duration.append(time_df['posix', 'max'][i] - time_df['posix', 'min'][i])

  df['extrafeatures'] = df.apply(lambda row: extrafeatures33(
      row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2), axis=1)

  df_median = df.groupby('extrafeatures').median()
  df_median.reset_index(inplace=True)
  df_median = df_median.add_suffix('-median')
  df_median.rename(columns={'extrafeatures-median': 'Phase'}, inplace=True)

  df_iqr = df.groupby('extrafeatures').agg(iqr)
  df_iqr.reset_index(inplace=True)
  df_iqr = df_iqr.add_suffix('-iqr')
  df_iqr.rename(columns={'extrafeatures-iqr': 'Phase'}, inplace=True)

  comb_df = pd.merge(df_median, df_iqr, on='Phase')
  comb_df['duration[s]'] = duration

  s = comb_df.set_index('Phase').stack()
  s.index = [f"{x}_{y}" for x, y in s.index]

  out = s.to_frame('').T

  out = out[out.columns.drop(list(out.filter(regex='Drop')))]

  ci = re.findall(r'CI_(\d+)', name)
  mpl = re.findall(r"MPL_(\d+.\d)", name)

  out['Cost_Index'] = ci
  out['Max_Payload'] = mpl

  return out


def datastore_prep(path, csv_name):

  directory_files = os.listdir(path)
  out_df = pd.DataFrame()

  for file_name in directory_files:

    tmp = data_prep(path + '/' + file_name)
    out_df = out_df.append(tmp, ignore_index=True)

    out_df.to_csv(csv_name)


path = '/home/gradguest/gradguest_data/PprVariables/PprTrainSet'

datastore_prep(path, 'PprTrain.csv')
