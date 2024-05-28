#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno


# In[2]:


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# In[3]:


data=pd.read_csv('latest_delhi.csv')


# #### DATA PREPROCESSING

# In[4]:


data.info()


# ##### CHECKING FOR MISSING VALUES AND HANDLING IT

# In[5]:


data.isnull().sum()


# In[6]:


### CHECKING THE PERCENTAGE OF MISSING VALUES IN EACH VARIABLE

data.isnull().sum().sort_values(ascending=False)/data.shape[0]


# In[7]:


msno.matrix(data)


# In[8]:


msno.bar(data)


# In[9]:


# Converting the 'Timestamp' column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')

# Set the 'Timestamp' column as the index as it is a time series data
#data.set_index('Timestamp', inplace=True)

# Inspect the first few rows to understand the dataset structure
data.head()


# In[10]:


data_eda=data.copy()
pollutants = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)',
                   'NH3 (µg/m³)', 'SO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)'
                   ]

# Setting the 'Timestamp' column as the index as it is a time series data
data_eda.set_index('Timestamp', inplace=True)
# Defining the layout of subplots
num_rows = len(pollutants)
num_cols = 1

# Creating subplots with the specified layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))  

# Plotting line charts for each column against years
for i, column in enumerate(pollutants):
    mean_by_year = data_eda.groupby(data_eda.index.year)[column].mean()
    years = mean_by_year.index
    values = mean_by_year.values
    
    ax = axes[i]  
    ax.plot(years, values, marker='s', color='green', linestyle='--')  
    ax.set_title(f'Yearly Average of {column} ')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Mean {column}')
    ax.grid(True)
    ax.set_xticks(years)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()  
plt.show()


# In[11]:


# Group by month and plot
fig, axes = plt.subplots(len(pollutants), 1, figsize=(10, 5 * len(pollutants)))

for i, column in enumerate(pollutants):
    mean_by_month = data_eda.groupby(data_eda.index.month)[column].mean()
    months = mean_by_month.index
    values = mean_by_month.values
    
    ax = axes[i]
    ax.plot(months, values, marker='s', color='green', linestyle='--')
    ax.set_title(f'Monthly Average of {column}')
    ax.set_xlabel('Month')
    ax.set_ylabel(f'Mean {column}')
    ax.grid(True)
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[12]:


# Create a list of years present in the data
years = data_eda.index.year.unique()

# Defining the layout of subplots
num_rows = len(pollutants)
num_cols = 1

# Creating subplots with the specified layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))

# Check if there's only one row; if so, wrap axes in a list
if num_rows == 1:
    axes = [axes]

# Plotting line charts for each column, with one line per year
for i, column in enumerate(pollutants):
    ax = axes[i]
    
    # Plot each year in a different line
    for year in years:
        # Select the data for the year
        yearly_data = data_eda[data_eda.index.year == year]
        
        # Group by month within this year
        mean_by_month = yearly_data.groupby(yearly_data.index.month)[column].mean()
        months = mean_by_month.index
        values = mean_by_month.values
        
        # Plotting
        ax.plot(months, values, marker='s', linestyle='--', label=str(year))
    
    ax.set_title(f'Monthly Average of {column} by Year')
    ax.set_xlabel('Month')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylabel(f'Mean {column}')
    ax.legend(title='Year')
    ax.grid(True)

plt.tight_layout()
plt.show()


# In[13]:


# Group by hour and plot
fig, axes = plt.subplots(len(pollutants), 1, figsize=(10, 5 * len(pollutants)))

for i, column in enumerate(pollutants):
    mean_by_hour = data_eda.groupby(data_eda.index.hour)[column].mean()
    hours = mean_by_hour.index
    values = mean_by_hour.values
    
    ax = axes[i]
    ax.plot(hours, values, marker='s', color='green', linestyle='--')
    ax.set_title(f'Hourly Average of {column}')
    ax.set_xlabel('Hour')
    ax.set_ylabel(f'Mean {column}')
    ax.grid(True)
    ax.set_xticks(hours)
    ax.tick_params(axis='x', rotation=360)

plt.tight_layout()
plt.show()


# In[ ]:





# In[14]:


# Group by day and plot
fig, axes = plt.subplots(len(pollutants), 1, figsize=(10, 5 * len(pollutants)))

for i, column in enumerate(pollutants):
    mean_by_day = data_eda.groupby(data_eda.index.day)[column].mean()
    days = mean_by_day.index
    values = mean_by_day.values
    
    ax = axes[i]
    ax.plot(days, values, marker='s', color='green', linestyle='--')
    ax.set_title(f'Daily Average of {column}')
    ax.set_xlabel('Day')
    ax.set_ylabel(f'Mean {column}')
    ax.grid(True)
    ax.set_xticks(days)
    ax.tick_params(axis='x', rotation=360)

plt.tight_layout()
plt.show()


# In[15]:


# Pandas day of the week is 0=Monday, to start from Sunday, we add 1 and take modulo 7
data_eda['DayOfWeek'] = (data_eda.index.dayofweek + 1) % 7  
fig, axes = plt.subplots(len(pollutants), 1, figsize=(10, 5 * len(pollutants)))

for i, column in enumerate(pollutants):
    mean_by_dow = data_eda.groupby('DayOfWeek')[column].mean()
    dow = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    
    ax = axes[i]
    ax.plot(dow, mean_by_dow.loc[[6, 0, 1, 2, 3, 4, 5]].values, marker='s', color='green', linestyle='--')  # Reorder to start from Sunday
    ax.set_title(f'Average of {column} by Day of the Week')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel(f'Mean {column}')
    ax.grid(True)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[16]:


### Distribution of Weather Data


# In[ ]:





# In[17]:


### Description of the dataframe before handling the missing values
data.describe()


# In[18]:


data.dtypes


# In[19]:


data.isna().sum()


# In[23]:


# Interpolating using a linear method
interpolated_data = data.interpolate(method='linear')

# Check for missing values after interpolation
missing_values_after = interpolated_data.isnull().sum()

missing_values_after


# In[21]:


interpolated_data.head()


# In[22]:


### Description of the dataframe after handling the missing values
interpolated_data.describe()


# In[24]:


interpolated_data.columns


# In[25]:


# Checking for duplicate rows
duplicate_rows = interpolated_data[interpolated_data.duplicated()]

# Print duplicate rows
print("Duplicate Rows:", duplicate_rows.shape)


# In[26]:


data_1=interpolated_data.copy()
data_1.head(10)


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Define the variables of interest for pollutants and weather data
pollutants = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)', 'NH3 (µg/m³)', 'SO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)' ]
weather_data = ['temp','dew', 'humidity','windspeed', 'winddir','sealevelpressure','solarradiation']

plt.figure(figsize=(18, 36))  # Adjust the figure size as needed

# Histograms
for i, variable in enumerate(pollutants + weather_data, 1):
    plt.subplot(9, 2, i)  # 9 rows and 2 columns for the subplot grid
    sns.histplot(data_1[variable], kde=True, color='blue')
    plt.title(f'Distribution of {variable}')
plt.tight_layout()
plt.show()

# Box Plots
plt.figure(figsize=(18, 36))  # Adjust the figure size as needed
for i, variable in enumerate(pollutants + weather_data, 1):
    plt.subplot(9, 2, i)  # 9 rows and 2 columns for the subplot grid
    sns.boxplot(y=data_1[variable])
    plt.title(f'Box Plot of {variable}')
plt.tight_layout()
plt.show()


# In[28]:


## Checking the Distribution of the pollutants
import matplotlib.pyplot as plt
import seaborn as sns

pollutants = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)', 'NH3 (µg/m³)', 'SO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)']

for pollutant in pollutants:
    plt.figure(figsize=(10, 4))
   
    sns.histplot(data_1[pollutant], kde=True, color='blue')
    plt.title(f'Distribution of {pollutant}')

    
    plt.show()


# In[29]:


weather_data = ['temp','dew', 'humidity','windspeed', 'winddir','sealevelpressure','solarradiation']
## Checking the Distribution of the weather conditions
import matplotlib.pyplot as plt
import seaborn as sns

for weather in weather_data:
    plt.figure(figsize=(10, 4))
   
    sns.histplot(data_1[weather], kde=True, color='green')
    plt.title(f'Distribution of {weather}')

    
    plt.show()
    


# #### Detecting Outliers

# In[31]:


# Obtaining Outliers using IQR for PM10
Q1 = data_1['PM10 (µg/m³)'].quantile(0.25)
Q3 = data_1['PM10 (µg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_pm10 = data_1['PM10 (µg/m³)'][(data_1['PM10 (µg/m³)']< lower_bound) | data_1['PM10 (µg/m³)'] > upper_bound]

print("The Outliers for PM10 (µg/m³) :")
print(outliers_pm10)

# Obtaining Outliers using IQR for PM2.5
Q1 = data_1['PM2.5 (µg/m³)'].quantile(0.25)
Q3 = data_1['PM2.5 (µg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_pm25 = data_1['PM2.5 (µg/m³)'][(data_1['PM2.5 (µg/m³)']< lower_bound) | data_1['PM2.5 (µg/m³)'] > upper_bound]

print("The Outliers for 'PM2.5 (µg/m³) :")
print(outliers_pm25)

# Obtaining Outliers using IQR for  NO2
Q1 = data_1['NO2 (µg/m³)'].quantile(0.25)
Q3 = data_1['NO2 (µg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_NO2 = data_1['NO2 (µg/m³)'][(data_1['NO2 (µg/m³)']< lower_bound) | data_1['NO2 (µg/m³)'] > upper_bound]

print("The Outliers for NO2 (µg/m³) :")
print(outliers_NO2)


# Obtaining Outliers using IQR for  NO
Q1 = data_1['NO (µg/m³)'].quantile(0.25)
Q3 = data_1['NO (µg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_NO = data_1['NO (µg/m³)'][(data_1['NO (µg/m³)']< lower_bound) | data_1['NO (µg/m³)'] > upper_bound]

print("The Outliers for NO (µg/m³) :")
print(outliers_NO)


# Obtaining Outliers using IQR for NH3 (µg/m³)
Q1 = data_1['NH3 (µg/m³)'].quantile(0.25)
Q3 = data_1['NH3 (µg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_NH3 = data_1['NH3 (µg/m³)'][(data_1['NH3 (µg/m³)']< lower_bound) | data_1['NH3 (µg/m³)'] > upper_bound]

print("The Outliers for NH3 (µg/m³) :")
print(outliers_NH3)

# Obtaining Outliers using IQR for NH3 (µg/m³)
Q1 = data_1['SO2 (µg/m³)'].quantile(0.25)
Q3 = data_1['SO2 (µg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_SO2 = data_1['SO2 (µg/m³)'][(data_1['SO2 (µg/m³)']< lower_bound) | data_1['SO2 (µg/m³)'] > upper_bound]

print("The Outliers for SO2 (µg/m³) :")
print(outliers_SO2)

# Obtaining Outliers using IQR for CO
Q1 = data_1['CO (mg/m³)'].quantile(0.25)
Q3 = data_1['CO (mg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_CO = data_1['CO (mg/m³)'][(data_1['CO (mg/m³)']< lower_bound) | data_1['CO (mg/m³)'] > upper_bound]

print("The Outliers for CO (mg/m³) :")
print(outliers_CO)


# Obtaining Outliers using IQR for Ozone
Q1 = data_1['Ozone (µg/m³)'].quantile(0.25)
Q3 = data_1['Ozone (µg/m³)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_Ozone = data_1['Ozone (µg/m³)'][(data_1['Ozone (µg/m³)']< lower_bound) | data_1['Ozone (µg/m³)'] > upper_bound]

print("The Outliers for Ozone (µg/m³) :")
print(outliers_Ozone)



# ### FEATURE ENGINEERING WITH MOVING AVERAGE

# In[32]:


rolling_data=data_1.copy()


# For Ozone 
rolling_data['Ozone_8hr_avg'] = rolling_data['Ozone (µg/m³)'].rolling(window=8, min_periods=1).mean()

# For CO 
rolling_data['CO_8hr_avg'] = rolling_data['CO (mg/m³)'].rolling(window=8, min_periods=1).mean()

#FOR PM10
rolling_data['PM10_24hr_avg'] = rolling_data['PM10 (µg/m³)'].rolling(window=24, min_periods=16).mean()

#FOR PM2.5
rolling_data['PM25_24hr_avg'] = rolling_data['PM2.5 (µg/m³)'].rolling(window=24, min_periods=16).mean()

#FOR SO2
rolling_data['SO2_24hr_avg'] = rolling_data['SO2 (µg/m³)'].rolling(window=24, min_periods=16).mean()

#FOR NO2
rolling_data['NO2_24hr_avg'] = rolling_data['NO2 (µg/m³)'].rolling(window=24, min_periods=16).mean()

#FOR NH3
rolling_data['NH3_24hr_avg'] = rolling_data['NH3 (µg/m³)'].rolling(window=24, min_periods=16).mean()


# In[33]:


rolling_data.isna().sum()


# In[34]:


rolling_data_1=rolling_data.copy()
rolling_data_1=rolling_data_1.dropna()
rolling_data_1.isna().sum()


# ### AQI CALCULATION

# In[35]:


## PM10 Sub-Index calculation
def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0

rolling_data_1['PM10_subindex'] = rolling_data_1['PM10_24hr_avg'].apply(lambda x: get_PM10_subindex(x))


# In[36]:


## PM2.5 Sub-Index calculation
def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0

rolling_data_1['PM2.5_subindex'] = rolling_data_1['PM25_24hr_avg'].apply(lambda x: get_PM25_subindex(x))



# In[37]:


## SO2 Sub-Index calculation
def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

rolling_data_1['SO2_subindex'] = rolling_data_1['SO2_24hr_avg'].apply(lambda x: get_SO2_subindex(x))


# In[38]:


## NO2 Sub-Index calculation
def get_NO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

rolling_data_1["NO2_SubIndex"] = rolling_data_1["NO2_24hr_avg"].apply(lambda x: get_NO2_subindex(x))



# In[39]:


## NH3 Sub-Index calculation
def get_NH3_subindex(x):
    if x <= 200:
        return x * 50 / 200
    elif x <= 400:
        return 50 + (x - 200) * 50 / 200
    elif x <= 800:
        return 100 + (x - 400) * 100 / 400
    elif x <= 1200:
        return 200 + (x - 800) * 100 / 400
    elif x <= 1800:
        return 300 + (x - 1200) * 100 / 600
    elif x > 1800:
        return 400 + (x - 1800) * 100 / 600
    else:
        return 0

rolling_data_1["NH3_SubIndex"] = rolling_data_1["NH3_24hr_avg"].apply(lambda x: get_NH3_subindex(x))



# In[40]:


## CO Sub-Index calculation
def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

rolling_data_1["CO_SubIndex"] = rolling_data_1["CO_8hr_avg"].apply(lambda x: get_CO_subindex(x))



# In[41]:


## O3 Sub-Index calculation
def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0

rolling_data_1["O3_SubIndex"] = rolling_data_1["Ozone_8hr_avg"].apply(lambda x: get_O3_subindex(x))



# In[42]:


# Calculating the AQI as the maximum of the sub-indices
rolling_data_1['AQI'] = rolling_data_1[['PM2.5_subindex', 'PM10_subindex', 'SO2_subindex', 
                                                'NO2_SubIndex', 'NH3_SubIndex', 'CO_SubIndex', 'O3_SubIndex']].max(axis=1)


# In[43]:


# Identifying the pollutant responsible for the AQI value and removing '_subindex' from the result
rolling_data_1['Dominant_Pollutant'] = rolling_data_1[['PM2.5_subindex', 'PM10_subindex', 'SO2_subindex', 
                                                               'NO2_SubIndex', 'NH3_SubIndex', 'CO_SubIndex', 'O3_SubIndex'
                                                              ]].idxmax(axis=1).str.replace('_subindex', '', case=False)


# #### Further Feature Engineering

# In[44]:


data_2=rolling_data_1.copy()

# Extracting Time Components
data_2['Date'] = data_2['Timestamp'].dt.date
data_2['Time'] = data_2['Timestamp'].dt.time
data_2['Day'] = data_2['Timestamp'].dt.day
data_2['Month'] = data_2['Timestamp'].dt.month
data_2['Year'] = data_2['Timestamp'].dt.year
data_2['DayOfWeek'] = data_2['Timestamp'].dt.dayofweek    
data_2['DayName'] = data_2['Timestamp'].dt.day_name()  



# Show the head of the DataFrame to verify
data_2[['Timestamp', 'Date', 'Time', 'Day','DayOfWeek','DayName', 'Month', 'Year']].head()



# In[45]:


data_2 = data_2.reset_index(drop=True)

#Introducting Season
def month_to_season(month):
    
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5, 6]:
        return 'Summer'
    elif month in [7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'Autumn'

# Applying the function to the DataFrame to create a 'Season' column
data_2['Season'] = data_2['Month'].apply(month_to_season)

data_2.head()


# In[ ]:





# In[46]:


data_df=data_2.copy()

import pandas as pd
import matplotlib.pyplot as plt

data_df.set_index('Timestamp', inplace=True)  # Set the datetime column as the index

# Daily AQI Trends
plt.figure(figsize=(15,5))
data_df['AQI'].plot(title='Daily AQI Trends')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Seasonal Trends
seasonal_aqi = data_df.groupby('Season')['AQI'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='Season', y='AQI', data=seasonal_aqi, palette='hsv')
plt.title('Seasonal AQI Trends')
plt.xlabel('Season')
plt.ylabel('Average AQI')
plt.show()

# Yearly Comparison
yearly_aqi = data_df.groupby('Year')['AQI'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='Year', y='AQI', data=yearly_aqi, palette='hsv')
plt.title('Yearly AQI Trends')
plt.xlabel('Year')
plt.ylabel('Average AQI')
plt.show()



# In[47]:


import matplotlib.pyplot as plt


# Grouping by year and calculating the mean AQI for each year
mean_aqi_by_year = data_df.groupby(data_df.index.year)['AQI'].mean()

# Plotting the yearly AQI averages
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
mean_aqi_by_year.plot(marker='s', color='green', linestyle='--', title='Yearly Average of AQI')
plt.xlabel('Year')
plt.ylabel('Mean AQI')
plt.grid(True)
plt.xticks(rotation=45)
plt.xticks(ticks=range(data_df['Year'].min(), data_df['Year'].max() + 1))
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()



# In[48]:


import matplotlib.pyplot as plt


# Grouping by month and calculating the mean AQI for each month
mean_aqi_by_month = data_df.groupby(data_df.index.month)['AQI'].mean()

# Plotting the monthly AQI averages
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
mean_aqi_by_month.plot(marker='s', color='green', linestyle='--', title='Monthly Average of AQI')
plt.xlabel('Month')
plt.ylabel('Mean AQI')
plt.grid(True)
plt.xticks(ticks=range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[49]:


import matplotlib.pyplot as plt

# Group by day of the month and calculate the mean AQI for each day
mean_aqi_by_day = data_df.groupby(data_df.index.day)['AQI'].mean()

# Plotting the daily AQI averages
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
mean_aqi_by_day.plot(marker='s', color='green', linestyle='--', title='Daily Average of AQI')
plt.xlabel('Day of the Month')
plt.ylabel('Mean AQI')
plt.grid(True)
plt.xticks(ticks=range(1, 32), labels=range(1, 32))  # Set x-ticks to show every day of the month
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[50]:


import matplotlib.pyplot as plt


# Create a series for the mean AQI by day of the week
mean_aqi_by_dow = data_df.groupby('DayOfWeek')['AQI'].mean()

# Days of the week starting with Sunday
dow = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

# Start plotting
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
plt.plot(dow, mean_aqi_by_dow.loc[[6, 0, 1, 2, 3, 4, 5]].values, marker='s', color='green', linestyle='--')  # Reorder to start from Sunday
plt.title('Average AQI by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Mean AQI')
plt.grid(True)
plt.xticks(range(7), dow)
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[51]:


import matplotlib.pyplot as plt

# Get the unique years from the data to create a line for each
years = data_df.index.year.unique()

# Start plotting
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

# Plot each year in a different line
for year in years:
    # Select the data for the year
    yearly_data = data_df[data_df.index.year == year]
    
    # Group by month within this year
    mean_aqi_by_month = yearly_data.groupby(yearly_data.index.month)['AQI'].mean()
    months = mean_aqi_by_month.index
    values = mean_aqi_by_month.values
    
    # Plotting
    plt.plot(months, values, marker='s', linestyle='--', label=str(year))

# Finalize plot details
plt.title('Monthly Average of AQI by Year')
plt.xlabel('Month')
plt.xticks(ticks=range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.ylabel('Mean AQI')
plt.legend(title='Year')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[52]:


import matplotlib.pyplot as plt
dominantpollutant_aqi = data_df.groupby(['Year', 'Dominant_Pollutant'])['Dominant_Pollutant'].count()
# Create a DataFrame from your grouped data if it's not already in that form
dominant_pollutant_df = dominantpollutant_aqi.reset_index(name='Count')

# Pivot your data to get 'Year' as index and each 'Dominant_Pollutant' as a column
pollutant_pivot = dominant_pollutant_df.pivot(index='Year', columns='Dominant_Pollutant', values='Count')

# Plotting
pollutant_pivot.plot(kind='bar', stacked=True, figsize=(10, 7))

plt.title('Count of Dominant Pollutants for AQI by Year')
plt.xlabel('Year')
plt.ylabel('Count of Dominant Pollutant')
plt.legend(title='Dominant Pollutant')
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()


# In[53]:


# Import necessary library for plotting time series of AQI
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df.set_index('Date', inplace=True)

# Aggregating to daily AQI
daily_aqi = data_df['AQI'].resample('D').mean()

# Apply seasonal decomposition
decomposition = seasonal_decompose(daily_aqi, model='additive', period=30)

# Plot the decomposed components of the time series
plt.figure(figsize=(22, 15)) 
plt.suptitle('Daily AQI Decomposition', fontsize=22) 

# Plot the observed values
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed', color='blue')
plt.legend(loc='upper left')
plt.tick_params(axis='both', which='major', labelsize=12)

# Plot the trend component
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.tick_params(axis='both', which='major', labelsize=12)

# Plot the seasonal component
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')
plt.tick_params(axis='both', which='major', labelsize=12)

# Plot the residual component
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='black')
plt.legend(loc='upper left')
plt.tick_params(axis='both', which='major', labelsize=12)

# Adjust the layout to ensure there is no overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




