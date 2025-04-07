import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import optuna
import numpy as np
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.linear_model import Ridge


df = pd.read_csv('Dynamic Price Prediction\Book.csv')
df_booking = df.drop(columns={'Unnamed: 0'})
print(df_booking.info())
df_user = pd.read_excel('Datas\\userUrban.xlsx')
print(df_user.info())
df_user = df_user.rename(columns={'user_id':'User_ID'})
df = pd.merge(df_user, df_booking, on="User_ID", how="inner")
df = df.drop(columns={'name','gender','age','email','phone_number','preferred_services','booking_history_count',
                 'last_booking_date','feedback_score','Booking_ID','Service_Location','Booking_Status','Payment_Status'})

df["Booking_Date"] = pd.to_datetime(df["Booking_Date"], errors="coerce", format="%d-%m-%Y", dayfirst=True)
df["Service_Date"] = pd.to_datetime(df["Service_Date"], errors="coerce", format="%d-%m-%Y", dayfirst=True)

df['service_lead_time'] = (df['Service_Date'] - df['Booking_Date']).dt.days

df['Service_Time'] = pd.to_datetime(df['Service_Time'])
df["Service_hour"] = df["Service_Time"].dt.hour

# def categorize_time(hour):
#     if 6 <= hour < 12:
#         return "Morning"
#     elif 12 <= hour < 17:
#         return "Afternoon"
#     elif 17 <= hour < 22:
#         return "Evening"
#     else:
#         return "Night"

# df["Service_Time_Slot"] = df["Service_hour"].apply(categorize_time)

# encoder = OrdinalEncoder(categories=[['Morning','Afternoon','Evening','Night']])  # Define the order
# encoded_data = encoder.fit_transform(df[['Service_Time_Slot']])

# df['time_slots'] = encoded_data

print(df.info())
df = df.drop(columns={'User_ID','cancellation_rate','average_rating_given','wallet_balance','subscription_status','Service_Time','Booking_Time'})
X = df.drop(columns={'Price'}, axis=1)
y = df['Price']
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.select_dtypes(include=['float64', 'int32', 'int64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

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

print("Columns in X_train:", X_train.columns)
print("Target column name:", y_train.name)
print(X_train.info())

# a = df.select_dtypes(include=['float64','int64','int32'])
# print(df.info())

# corr_matrix = a.corr()

# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))

# # Create the heatmap
# sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap="coolwarm", linewidths=0.5)

# plt.show()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
    }

    print(f"Trial {trial.number} parameters: {params}")

    model = XGBRegressor(**params, random_state=42, eval_metric="rmse", n_jobs=-1)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error").mean()
    
    print(f"Trial {trial.number} completed with RMSE: {-cv_score:.4f}")

    return -cv_score  

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best Parameters:", best_params)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(**best_params, random_state=42, eval_metric="rmse", n_jobs=-1))
])

final_pipeline.fit(X_train, y_train)

y_pred_test = final_pipeline.predict(X_test)
y_pred_train = final_pipeline.predict(X_train)

                                                            
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

evaluate_model("Training Set", y_train, y_pred_train)
evaluate_model("Test Set", y_test, y_pred_test)


joblib.dump(final_pipeline, "Price_prediction_pipeline.pkl")
print("\nPipeline saved successfully!")


