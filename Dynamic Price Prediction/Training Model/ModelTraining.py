from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Database Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["UrbanCap"]
collection_booking = db["UrbanCap"]
collection_user = db["User_table"]

# Fetch Data
data_user = list(collection_user.find({}, {"_id": 0})) 
data_booking = list(collection_booking.find({}, {"_id": 0})) 

# Convert Data to DataFrame
df_user = pd.DataFrame(data_user) if data_user else None
df_booking = pd.DataFrame(data_booking) if data_booking else None

if df_user is not None:
    df_user = df_user.iloc[:, 1:].copy()
    df_user.insert(0, 'User_ID', range(len(df_user)))

if df_booking is not None:
    df_booking = df_booking.iloc[:, 1:].copy()


# print(df_booking.info())
# print(df_user.info())

df_booking['User_ID'] = df_booking['User_ID'].astype(int)

# Merge Data
df = pd.merge(df_user, df_booking, on='User_ID', how="inner") if df_user is not None and df_booking is not None else None

# Data type conversions
df['User_ID'] = df['User_ID'].astype(int)
df['Total_Amount'] = df['Total_Amount'].astype(float)
df['Discount_Applied'] = df['Discount_Applied'].astype(float)
df['Tax_Amount'] = df['Tax_Amount'].astype(float)
df['Final_Payment'] = df['Final_Payment'].astype(float)
df['Location_ID'] = df['Service_Location'].astype(int)

# Convert Date Columns
df["Booking_Date"] = pd.to_datetime(df["Booking_Date"], errors="coerce", format="%d-%m-%Y", dayfirst=True)
df["Service_Date"] = pd.to_datetime(df["Service_Date"], errors="coerce", dayfirst=True)

# Extract Date Features
df["Booking_hour"] = df["Booking_Date"].dt.hour
df["Booking_day_of_week"] = df["Booking_Date"].dt.dayofweek
df["Booking_Month"] = df["Booking_Date"].dt.month
df["Service_hour"] = df["Service_Date"].dt.hour
df["Service_day_of_week"] = df["Service_Date"].dt.dayofweek
df["Service_Month"] = df["Service_Date"].dt.month

# Feature Selection

x = pd.DataFrame(df, columns=['User_ID', 'Provider_ID', 'gender', 'Location_ID', 'preferred_services', 'preferred_time_slots',
        'booking_history_count', 'cancellation_rate', 'average_rating_given', 'wallet_balance',
        'subscription_status', 'Booking_Date', 'Booking_Time', 'Service_Date', 'Service_Time', 'Final_Payment'])
# x = df[['User_ID', 'Provider_ID', 'gender', 'Location_ID', 'preferred_services', 'preferred_time_slots',
#         'booking_history_count', 'cancellation_rate', 'average_rating_given', 'wallet_balance',
#         'subscription_status', 'Booking_Date', 'Booking_Time', 'Service_Date', 'Service_Time', 'Final_Payment']].copy()

# Handle Time Slots
time_slot_mapping = {
    'Morning': (6, 12),
    'Afternoon': (12, 17),
    'Evening': (17, 22),
    'Night': (22, 6)
}

x['preferred_time_range'] = x['preferred_time_slots'].map(lambda slot: time_slot_mapping.get(slot, (0, 24)))
x["Service_time"] = pd.to_datetime(x["Service_Time"])
x["Service_hour"] = x["Service_time"].dt.hour
# x["Service_time"] = pd.to_datetime(x["Service_Time"], errors="coerce", format="%H:%M")
x.drop(columns=['Service_time'], inplace=True)

# Handle Time Slot Matching
# x['preferred_time_start'] = x['preferred_time_range'].apply(lambda t: t[0])
# x['preferred_time_end'] = x['preferred_time_range'].apply(lambda t: t[1])
# x['time_slot_match'] = x.apply(lambda row: row['preferred_time_start'] <= row['Service_hour'] < row['preferred_time_end'], axis=1)
# x.drop(columns=['preferred_time_start', 'preferred_time_end'], inplace=True)
x[['preferred_time_start', 'preferred_time_end']] = x['preferred_time_range'].apply(lambda t: pd.Series(t))
x.drop(columns=['preferred_time_range'], inplace=True)


# Compute Service Lead Time
x['Booking_Date'] = pd.to_datetime(x['Booking_Date'])
x['Service_Date'] = pd.to_datetime(x['Service_Date'])
x['service_lead_time'] = (x['Service_Date'] - x['Booking_Date']).dt.days

# Fix categorical data types
x['Provider_ID'] = x['Provider_ID'].astype(int)
x["preferred_services"] = x["preferred_services"].astype(str)
x["preferred_time_slots"] = x["preferred_time_slots"].astype(str)

# Feature Selection
num_features = x.select_dtypes(include=['float64', 'int32', 'int64']).columns.tolist()
cat_features = x.select_dtypes(include=['bool', 'object']).columns.tolist()

# Handle Missing Data
# x['Service_hour'].fillna(-1, inplace=True)

print(x.info())
print(x)

x = pd.DataFrame(x)
# Preprocessing Pipelines
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
]) if cat_features else 'passthrough'

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

x = pd.DataFrame(x)
print("wow")

# x[['preferred_time_start', 'preferred_time_end']] = x['preferred_time_range'].apply(lambda t: pd.Series(t))
# x.drop(columns=['preferred_time_range'], inplace=True)


x = pd.DataFrame(x, columns=['User_ID', 'Provider_ID', 'gender', 'Location_ID', 'preferred_services', 'preferred_time_slots',
        'booking_history_count', 'cancellation_rate', 'average_rating_given', 'wallet_balance',
        'subscription_status', 'Booking_Date', 'Booking_Time', 'Service_Date', 'Service_Time',
        'preferred_time_start','preferred_time_end','Service_hour','time_slot_match','service_lead_time', 'Final_Payment'])

print("wow2")

print(x.dtypes)
print(x.head())

# Apply Transformation
transformed_x = preprocessor.fit_transform(x)
transformed_x = pd.DataFrame(preprocessor.fit_transform(x), columns=num_features + cat_features)

# # Print Summary
print(transformed_x.info())
print(transformed_x)

corr_matrix = transformed_x.corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Create the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5)

plt.show()