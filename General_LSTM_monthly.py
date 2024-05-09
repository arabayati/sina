import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import scipy.stats
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import random
import json

start_time = datetime.now()

# Parameters
TargetLabel = 'streamflow_mmd'
LearningRate = 0.001 # 0.001/0.0001
TIME_STEP = 365
EPOCHs = 75
BatchSize = 256 # 128/256/512/1024
Patience = 50
TrainRatio = 0.4
ValidationRatio = 0.2

# Input columns
f_columns =['mean_temperature_C', 'precipitation_mmd', 'pet_mmd']
staticColumns=['area_km2','mean_elevation_m','mean_slope_mkm','shallow_soil_hydc_md',
               'soil_hydc_md','soil_porosity','depth_to_bedrock_m','maximum_water_content_m',
               'bedrock_hydc_md','soil_bedrock_hydc_ratio','mean_precipitation_mmd',
               'mean_pet_mmd','aridity','snow_fraction','seasonality','high_P_freq_daysyear',
               'low_P_freq_daysyear','high_P_dur_day','low_P_dur_day','mean_forest_fraction_percent']

# Input folder (daily csv files)
folder = 'CAMELS-US/Daily_Data/'

# Output folder, where we save the results
outputfolder = 'CAMELS-US/Output_USCA/'

if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)
    print('Oops! directory did not exist, but no worries, I created it!')

SaveModel = outputfolder

#Static Data- it must contain items listed by "staticColumns" and grid code
path_static = 'CAMELS-US/Attributes/attributes.csv'

# Read and Normalize statistical features
dfs = pd.read_csv(path_static)  # Static Data
OurDesiredStaticAttributes = dfs.columns
f_transformer = StandardScaler()
f_transformer = f_transformer.fit(dfs[OurDesiredStaticAttributes].to_numpy())
dfs.loc[:, OurDesiredStaticAttributes] = f_transformer.transform(
  dfs[OurDesiredStaticAttributes].to_numpy()
)
dftemp = pd.read_csv(path_static)
dfs['gridcode'] = dftemp['gridcode'].astype(str)

# Create Dataset
def create_dataset(X, y, date_df, doy_df, time_steps=1):
    Xs, ys, date, doy = [], [], [], []
    for i in range(len(X) - time_steps):
        X_seq = X.iloc[i:(i + time_steps)]

        # Check if there's any NaN in the X sequence or the corresponding y value
        if not X_seq.isnull().values.any() and not pd.isnull(y.iloc[i + time_steps-1]):
            Xs.append(X_seq.values)
            ys.append(y.iloc[i + time_steps-1])
            date.append(date_df.iloc[i + time_steps-1])
            doy.append(doy_df.iloc[i + time_steps-1])

    return Xs, ys, np.array(date), np.array(doy)


# NSE function (Nash-Sutcliff-Efficiency)
def NSE(targets,predictions):
  return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))

# model definition
model = keras.Sequential()
model.add(keras.layers.LSTM(units=256, return_sequences=False, input_shape=(TIME_STEP, len(f_columns) + len(staticColumns))))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=1, activation='relu))

callbacks = [keras.callbacks.EarlyStopping(patience=Patience, restore_best_weights=True)]
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LearningRate))     # compile a model based on MSE

# count number of files
def count_files(directory):
    entries = os.listdir(directory)
    file_count = sum(os.path.isfile(os.path.join(directory, entry)) for entry in entries)
    return file_count

# Global data structure to store trained data
file_path = 'CAMELS-US/Output_USCA/used_months.json'
if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        used_months = json.load(file)
    print("JSON file already exists and read into RAM")
else:
    used_months = {}
    print("JSON file created for the first time")
    
# Load weights if restart training
h5_files = [f for f in os.listdir(SaveModel) if f.endswith('.h5')]
if h5_files:
    h5_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    latest_h5_file = os.path.join(SaveModel, h5_files[0])
    iteration = int(h5_files[0].split('_')[1]) + 1
    model.load_weights(latest_h5_file)
    print(f"Loaded weights from {latest_h5_file}")
else:
    iteration = 1
    print("No .h5 files found, starting training from scratch.")


useful_months = {}  # Store the usable months: total months - 12

# Read all csv files into RAM as dataframe
all_files = {}
for file in os.listdir(folder):
    filename = file.rstrip(".csv")
    Dir = folder + str(file)
    dataframe = pd.read_csv(Dir)
    all_files[filename] = dataframe
    all_files[filename]['date'] = pd.to_datetime(all_files[filename].pop('date'))
    useful_months[filename] = all_files[filename]['date'].dt.to_period('M').nunique() - 12
    
# Change based on RAM usage, num_train_months : num_val_months = 2 : 1
num_train_months = 2
num_val_months = int(num_train_months / 2)

# 6k/12k pages
#num_pages = 12000

# Change based on the number of useful months needed for training
iterations = int(187 / num_train_months)

#samples_df = pd.read_csv('80_Samples.csv', header=None, dtype=str)
#samples_list = samples_df[0].tolist()

# Training loop
while (iteration <= iterations):
    X_train, y_train = [], []
    X_val, y_val = [], []
    # X_test, y_test = [], []
    print(f"Iteration {iteration}/{iterations}")
    
    # Iterate over all the files
    for GridCode in all_files.keys():
    # for GridCode in samples_list:
        if GridCode not in used_months.keys():
            used_months[GridCode] = []

        # Grab one file based on GridCode
        df = all_files[GridCode].copy()
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        df[TargetLabel] = np.log1p(df[TargetLabel])
        
        # Randomly select `num_train_months + num_val_months` months
        possible_train_months = [i+1 for i in range(int(useful_months[GridCode] * TrainRatio)) if i+1 not in used_months[GridCode]]
        possible_val_months = [i+1 for i in range(int(useful_months[GridCode] * TrainRatio), int(useful_months[GridCode] * TrainRatio + useful_months[GridCode] * ValidationRatio)) if i+1 not in used_months[GridCode]]
        selected_train_months = random.sample(possible_train_months, num_train_months)
        selected_val_months = random.sample(possible_val_months, num_val_months)
        selected_months = selected_train_months + selected_val_months

        # For each month in a file, filter the corresponding pages
        for selected_month in selected_months:
            selected_year = int(selected_month / 12) + int(df['year'][0])
            selected_year_month = 12 if selected_month % 12 == 0 else selected_month % 12
            
            # Different months have different number of pages
            if selected_year_month in [1, 3, 5, 7, 8, 10, 12]:
                end_date_of_month = 31
            elif selected_year_month == 2: # For simplicity we don't consider leep year, this only loses 1page/4years data
                end_date_of_month = 28
            else:
                end_date_of_month = 30
                
            used_months[GridCode].append(selected_month)
            
            # Filtering the month region from file
            start_date = pd.Timestamp(year=selected_year, month=selected_year_month, day=1)
            end_date = pd.Timestamp(year=selected_year + 1, month=selected_year_month, day=end_date_of_month)
            monthly_pages_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

            # Input processing
            f_transformer = StandardScaler().fit(monthly_pages_df[f_columns])
            monthly_pages_df[f_columns] = f_transformer.transform(monthly_pages_df[f_columns])
            static_row = dfs[dfs['gridcode'] == GridCode]
            for item in staticColumns:
                monthly_pages_df.loc[:, item] = static_row[item].to_numpy()[0]

            # Creating pages for month
            input_columns = f_columns + staticColumns
            X, y, train_date, train_days = create_dataset(monthly_pages_df[input_columns], monthly_pages_df[TargetLabel], monthly_pages_df['date'], monthly_pages_df['day_of_year'], time_steps=TIME_STEP)
        
            if selected_month in selected_train_months:
                X_train.extend(X)
                y_train.extend(y)
            elif selected_month in selected_val_months:
                X_val.extend(X)
                y_val.extend(y)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    # print(X_train.shape, y_train.shape)
    # print(X_val.shape, y_val.shape)
    # X_test, y_test = np.array(X_test), np.array(y_test)

    # num_batches = round(X_train.shape[0] / num_pages)
    
    # indices_train = np.arange(X_train.shape[0])
    # indices_val = np.arange(X_val.shape[0])
    # np.random.shuffle(indices_train)
    # np.random.shuffle(indices_val)

    # X_train = X_train[indices_train]
    # y_train = y_train[indices_train]
    # X_val = X_val[indices_val]
    # y_val = y_val[indices_val]

    # X_train_batches = np.array_split(X_train, num_batches)
    # y_train_batches = np.array_split(y_train, num_batches)
    # X_val_batches = np.array_split(X_val, num_batches)
    # y_val_batches = np.array_split(y_val, num_batches)

    # for X_train_batch, y_train_batch, X_val_batch, y_val_batch in zip(X_train_batches, y_train_batches, X_val_batches, y_val_batches):
    #     history = model.fit(
    #         X_train_batch, y_train_batch,
    #         epochs=EPOCHs,
    #         batch_size=BatchSize,
    #         validation_data=(X_val_batch, y_val_batch),
    #         shuffle=True,
    #         callbacks=callbacks
    #         )

    history = model.fit(
            X_train, y_train,
            epochs=EPOCHs,
            batch_size=BatchSize,
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=callbacks
            )
    with open(file_path, 'w') as f:
        json.dump(used_months, f)

    # np.savetxt(SaveModel + 'Pages_Based_On_Which_Trained_sofar.out', used_months, delimiter=',')
    path = SaveModel + 'interationNum_' + str(int(iteration))+'_Generally_Trained_UP_TO_NOW_Model' + '.h5'
    model.save_weights(path)

    iteration += 1
    
path = SaveModel + 'Generally_Trained_Model' +'.h5'
model.save_weights(path)

end_time = datetime.now()
duration = end_time - start_time
print(f"Model started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total duration: {duration}")
