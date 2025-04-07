import pandas as pd
import numpy as np

# Load data
df_booking = pd.read_csv("Datas\Booking.csv")
df_user = pd.read_excel("Datas\\userUrban.xlsx")

# import pandas as pd
# import numpy as np

# # Load Data
# df_booking = pd.read_csv("Booking.csv")
# df_user = pd.read_excel("userUrban.xlsx")

# Identify non-numeric values in numeric columns
def check_non_numeric(df):
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n=== Unique Values in {col} ===")
        print(df[col].unique()[:10])  # Show first 10 unique values for inspection

check_non_numeric(df_booking)

# Convert numeric columns with potential issues
numeric_cols = ["Total_Amount", "Discount_Applied", "Tax_Amount", "Final_Payment"]
for col in numeric_cols:
    df_booking[col] = pd.to_numeric(df_booking[col], errors='coerce')  # Convert errors to NaN

# Drop any remaining rows with NaN in key numeric columns
df_booking.dropna(subset=numeric_cols, inplace=True)

# Now try correlation matrix
print("\n=== Correlation Matrix (Booking Dataset) ===")
print(df_booking.corr())
