import pandas as pd
from pymongo import MongoClient
import joblib
import numpy as np

client = MongoClient("mongodb://localhost:27017/")
db = client["UrbanCap"]
collection_booking = db["Past Booking"]
collection_user = db["User_table"]

data_booking = list(collection_booking.find({}, {"_id": 0})) 
data_user = list(collection_user.find({}, {"_id": 0})) 

df_booking = pd.DataFrame(data_booking) if data_booking else None
df_user = pd.DataFrame(data_user) if data_user else None

if df_booking is not None:
    df_booking = df_booking.iloc[:, 1:].copy()

if df_user is not None:
    df_user = df_user.iloc[:, 1:].copy()
    df_user.insert(0, 'User_ID', range(len(df_user)))

df_booking['User_ID'] = df_booking['User_ID'].astype(int)


df = pd.merge(df_user, df_booking, on='User_ID', how="inner") if df_user is not None and df_booking is not None else None

df = df.drop(columns={'Price'})

df = df.drop(columns={'name','gender','age','email','phone_number','preferred_services','booking_history_count'
                      ,'cancellation_rate','average_rating_given','wallet_balance','last_booking_date','subscription_status'
                      ,'Service_Location','feedback_score','Booking_ID','User_ID','Booking_Status','Payment_Status','Booking_Time'})


'''
'location_id', 'preferred_time_slots', 'Provider_ID', 'Service_ID',
       'Booking_Date', 'Service_Date', 'service_lead_time', 'Service_hour' 
'''

df["Booking_Date"] = pd.to_datetime(df["Booking_Date"], errors="coerce", format="%d-%m-%Y", dayfirst=True)
df["Service_Date"] = pd.to_datetime(df["Service_Date"], errors="coerce", format="%d-%m-%Y", dayfirst=True)

df['service_lead_time'] = (df['Service_Date'] - df['Booking_Date']).dt.days

df['Service_Time'] = pd.to_datetime(df['Service_Time'])
df["Service_hour"] = df["Service_Time"].dt.hour

df = df.drop(columns=['Service_Time'])

pipeline = joblib.load("Price_prediction_pipeline.pkl")
def predict_price(data):
    """Uses trained pipeline to predict flight prices."""
    try:
        predictions = pipeline.predict(data)
        return np.round(predictions, 2)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    
feature_columns = df.drop(columns=["Actual_Price"], errors="ignore") 

df["Predicted_Price"] = predict_price(feature_columns)
# df = df.sort_values('')
print(df.sort_values(['Service_Date']))