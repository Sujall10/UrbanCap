import pandas as pd
from pymongo import MongoClient
import joblib
import numpy as np

client = MongoClient("mongodb://localhost:27017/")
db = client["UrbanCap"]
collection_booking = db["Past Booking"]


data_booking = list(collection_booking.find({}, {"_id": 0})) 

df_booking = pd.DataFrame(data_booking) if data_booking else None

if df_booking is not None:
    df_booking = df_booking.iloc[:, 1:].copy()

df_booking['Service_Date'] = pd.to_datetime(df_booking['Service_Date'])
print(df_booking.sort_values('Service_Date'))