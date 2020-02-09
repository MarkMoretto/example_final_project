#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Taxi data analysis
Date created: Sat Feb  8 07:51:28 2020

URI: https://github.com/EricSchles/example_final_project

Using January 2014 green taxi data:
    https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2014-01.csv
"""

import re
import gc
from io import StringIO
import urllib.request as ureq
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
gc.enable()
sns.set(style="whitegrid")
#sns.set(style="white", color_codes=True)


data_yr = 2014
data_mth = 1
uri = f"https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_{data_yr}-{data_mth:0>2}.csv"

### Read and decode raw data.  The size of this particular data set is ~118 MB
with ureq.urlopen(uri) as resp:
    raw_data = resp.read().decode("utf-8")
    
### The data is a little messy, so we'll replace the line separators with
### something a bit more consistent.  Then, we'll split the data into a 
### list of lines using our new separator.
raw_data_2 = re.split(r"\n", re.sub(r"\s*?\r\n|\s*?\n\n*?", "\n", raw_data))

### Columns are the first row (0-indexed) of our new data set
cols_ = raw_data_2[0].split(",")

### We'll concatenate everything besides the header row back into a single
### string with \n as our separator.
raw_data_3 = '\n'.join([i for i in raw_data_2[1:]])

### Because the column count of 20 doesn't match the split row count of 22,
### we'll join our values again, but leave off the last two characters in the row.
### We're using a StringIO buffer here to help the process out as it is easier
### for pandas to parse.
data_sio = StringIO('\n'.join([i[:-2] for i in raw_data_2[1:]]))

### Finally, we're creating our dataframe with a few explicit arguments passed.
data = pd.read_csv(data_sio, index_col = None, sep=",", header = None, names = cols_)

data_sio.close()
del raw_data, raw_data_2, raw_data_3, data_sio

### Print a row-column count to be sure that the data loaded properly.
print(f"The data contains:\n\t{data.shape[0]:,.0f} columns\n\t{data.shape[1]:,.0f} rows")

##################################################
########## Exploration and outliers ##############
"""
Evaluate null counts for each field. Drop any fields which have null counts
above a given threshold.
Ex. - If threshold_pct = 1.0, then all fields will be kept
      If 0.0, then all fields will be dropped
"""

threshold_pct = 0.10

df_null_pct = data.isnull().sum() / data.shape[0]
drop_cols = df_null_pct[df_null_pct > threshold_pct].index.values

### Create new dataframe with appro
df = data.drop(drop_cols, axis = 1).copy()

#### Add line column (maybe)
#df['line'] = 1

### Convert our datetime columns to the appropriate format
df_cols = df.columns.values
for col in df_cols:
    if "datetime" in col:
        df[col] = pd.to_datetime(df[col], infer_datetime_format = True)

### Make non-ID or non-foreign key fields float
df.loc[:, 'Passenger_count'] = df.loc[:, 'Passenger_count'].astype(float)


# df.loc[df[abs(stats.zscore(df['Passenger_count'])) < 3].index.values, 'Passenger_count']
# tmp = df.loc[df[abs(stats.zscore(df['Passenger_count'])) >= 3].index.values, 'Passenger_count']

### Quick error and/or outlier analysis
df_desc = df[['Fare_amount', 'Total_amount']].describe()



### RateCodeID metrics
rate_code_dict = {
    1: "Standard rate",
    2: "JFK",
    3: "Newark",
    4: "Nassau or Westchester",
    5: "Negotiated fare",
    6: "Group ride",
    99: None,
    }
rt_cd_cnt = df['RateCodeID'].value_counts()

### Map our dictionary values to index values
rt_cd_cnt.index = rt_cd_cnt.index.map(rate_code_dict)



### Payment Type
#df['Payment_type'].unique()
pmt_type_dict = {
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
    }
pmt_type_cnt = df['Payment_type'].value_counts()
pmt_type_cnt.index = pmt_type_cnt.index.map(pmt_type_dict)

# Look at `no charge` or `dispute` data
df2 = df.loc[((df['Payment_type'] == 3) | (df['Payment_type'] == 4)), :]
df2_desc = df2[['Trip_distance', 'Fare_amount', 'Extra','MTA_tax', 'Tip_amount', 'Tolls_amount', 'Total_amount',]].describe()


### Lat (x-axis) and long (y-axis) distances using manhattan distance
def manhattan(X_1, y_1, X_2, y_2):
    return abs(X_1 - X_2) + abs(y_1 - y_2)

lat_long_cols = [i for i in df.columns if i.endswith(('longitude', 'latitude',))]
df_ll_desc = df[lat_long_cols].describe()
df['manhattan_dist'] = manhattan(
                    df['Pickup_latitude'],
                    df['Pickup_longitude'],
                    df['Dropoff_latitude'],
                    df['Dropoff_longitude']
                    )

### Note descriptive stats.  
dist_desc = df[['Trip_distance','manhattan_dist']].describe()

### Outliers by quantile
q_dist_low, q_dist_hi = df["manhattan_dist"].quantile(0.01), df["manhattan_dist"].quantile(0.99)
df3 = df[(df["manhattan_dist"] > q_dist_low) & (df["manhattan_dist"] < q_dist_hi)]







### Fare_amount 
# Looking at descriptive stats for fare_amount, we can see that the minimum
# fare is below zero and the maximum fare is nearly 3000
#ax = sns.distplot(df['Fare_amount'])
df_fa_mean_ = df_desc.loc['mean', 'Fare_amount']
df_fa_stdev_ = df_desc.loc['std', 'Fare_amount']
fare_amount_upper_sd3 = df_fa_mean_ + df_fa_stdev_ * 3
fare_amount_lower_sd3 = df_fa_mean_ - df_fa_stdev_ * 3
df_fare_amt = df.loc[((df['Fare_amount'] >= fare_amount_lower_sd3) & (df['Fare_amount'] <= fare_amount_upper_sd3)), "Fare_amount"]

fig = plt.figure(figsize=(9, 7))
ax1 = sns.distplot(df_fare_amt)
ax1.set_title("Fare amount within 3 sd")
ax1.set_xlabel("Bins")
ax1.set_ylabel("%-age")
plt.tight_layout()
plt.show()




#### Trip distance was zero, what are passenger count and fare amount?
#df_ztd = df.loc[df['Trip_distance'] <= 0., ['Passenger_count','Fare_amount']]
#ztd_desc = df_ztd.describe()
#
#df_ztd = (df_ztd.sort_values(by=['Fare_amount', 'Passenger_count'], ascending=[False, False]).copy())
#df_ztd.head(15)
 


### Datetime difference
# (df.loc[:15, ['lpep_pickup_datetime', 'Lpep_dropoff_datetime']].apply(lambda x: (x['Lpep_dropoff_datetime'] - x['lpep_pickup_datetime']).seconds, axis = 1))

### Pickup to dropoff minutes
df['pu_do_minutes'] = round((df['Lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.seconds / 60., 4)
df['pu_do_minutes'].describe()





# https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
### Outliers by quantile
q_low = df["pu_do_minutes"].quantile(0.01)
q_hi  = df["pu_do_minutes"].quantile(0.99)
df_pu_do_min = df[(df["pu_do_minutes"] < q_hi) & (df["pu_do_minutes"] > q_low)]













