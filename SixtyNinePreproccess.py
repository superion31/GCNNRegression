#importing libraries
import pandas as pd
import numpy as np
from scipy.stats import iqr
import re
import os

#function for general categorization


def extrafeatures(row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2):
  if row['Flight_Block'] == 'climb' and row['h[ft]'] < cli_lim1:
    return 'Climbing(-)'
  elif row['Flight_Block'] == 'climb' and row['h[ft]'] >= cli_lim1 and row['h[ft]'] < cli_lim2:
    return 'Climbing'
  elif row['Flight_Block'] == 'climb' and row['h[ft]'] >= cli_lim2:
    return 'Climbing(+)'
  elif row['Flight_Block'] == 'cruise' and row['h[ft]'] < cru_lim1:
    return 'Cruising(-)'
  elif row['Flight_Block'] == 'cruise' and row['h[ft]'] >= cru_lim1 and row['h[ft]'] < cru_lim2:
    return 'Crusing'
  elif row['Flight_Block'] == 'cruise' and row['h[ft]'] >= cru_lim2:
    return 'Cruising(+)'
  elif row['Flight_Block'] == 'descent' and row['h[ft]'] < desc_lim1:
    return 'Descending(-)'
  elif row['Flight_Block'] == 'descent' and row['h[ft]'] >= desc_lim1 and row['h[ft]'] < desc_lim2:
    return 'Descending'
  elif row['Flight_Block'] == 'descent' and row['h[ft]'] >= desc_lim2:
    return 'Descending(+)'

#final concatenated dataframe


def data_prep(name):

  pre_df = pd.read_csv(name)
  pre_df = pre_df.loc[:, ~pre_df.columns.str.startswith('reserved')]
  pre_df = pre_df.replace(to_replace='CRUISE_.*', value='CRUISE', regex=True)

  time_keep = ['Flight_Block', 'Phase', 'posix[s]', 'h[ft]']
  time_df = pre_df[time_keep]
  duration = []

  to_keep = ['Flight_Block', 'Phase', 'h[ft]', 'Temp[ºC]', 'Press[hPa]', 'Wn[kt]',
             'We[kt]',  'Ws[kt]',  'Wx[kt]', 'Lat[º]', 'Lon[º]', 'vdot[kt/s]', 'hdot[ft/min]']

  df = pre_df[to_keep]

  delta_cli = (df[df['Flight_Block'] == 'climb']['h[ft]'].max()
               - df[df['Flight_Block'] == 'climb']['h[ft]'].min())/3
  cli_lim1 = df[df['Flight_Block'] == 'climb']['h[ft]'].min() + \
      delta_cli
  cli_lim2 = df[df['Flight_Block'] == 'climb']['h[ft]'].min() + \
      2*delta_cli

  delta_cru = (df[df['Flight_Block'] == 'cruise']['h[ft]'].max()
               - df[df['Flight_Block'] == 'cruise']['h[ft]'].min())/3
  cru_lim1 = df[df['Flight_Block'] == 'cruise']['h[ft]'].min() + delta_cru
  cru_lim2 = df[df['Flight_Block'] == 'cruise']['h[ft]'].min() + 2 * delta_cru

  delta_desc = (df[df['Flight_Block'] == 'descent']['h[ft]'].max()
                - df[df['Flight_Block'] == 'descent']['h[ft]'].min())/3
  desc_lim1 = df[df['Flight_Block']
                 == 'descent']['h[ft]'].min() + delta_desc
  desc_lim2 = df[df['Flight_Block']
                 == 'descent']['h[ft]'].min() + 2 * delta_desc

  df_median = df.copy()
  df_iqr = df.copy()

  time_df['extrafeatures'] = time_df.apply(
    lambda row: extrafeatures(row, cli_lim1, cli_lim2, cru_lim1, cru_lim2, desc_lim1, desc_lim2), axis=1)
  time_df = time_df.groupby('extrafeatures').agg(['min', 'max'])
  for i in range(len(time_df)):
      duration.append(time_df['posix[s]', 'max'][i]
                      - time_df['posix[s]', 'min'][i])

  df['extrafeatures'] = df.apply(lambda row: extrafeatures(
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
      print(len(out_df))

  out_df.to_csv(csv_name)


path = '/home/gradguest/gradguest_data/OrgnVariables/OrgnTrainSet'

datastore_prep(path, 'OrgnTrain.csv')
