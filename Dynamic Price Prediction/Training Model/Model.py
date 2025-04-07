import csv
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["UrbanCap"]
# collection = db["UrbanCap"]
collection = db['Past Booking']

# csv_file = "Past_Booking.csv"

csv_file = "Dynamic Price Prediction\Book.csv"

df = pd.read_csv(csv_file)

# with open(df, mode="r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)  # Convert CSV rows into dictionaries
#     data = list(reader)  # Convert iterator to a list of dictionaries

data = df.to_dict(orient="records")

# Insert data into MongoDB
collection.insert_many(data)

print("Data imported successfully!")
