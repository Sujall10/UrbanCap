from pymongo import MongoClient
import pandas as pd
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder


def load_data_from_mongo():
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["UrbanCap"]
    collection_booking = db["UrbanCap"]
    collection_user = db["User_table"]

    data_user = list(collection_user.find({}, {"_id": 0}))
    data_booking = list(collection_booking.find({}, {"_id": 0}))

    if data_user:
        df_user = pd.DataFrame(data_user)
        df_user = df_user.iloc[:, 1:]
        df_user.insert(0, 'User_ID', range(0, len(df_user)))
    else:
        print("No User data found in MongoDB!")
        df_user = pd.DataFrame()

    if data_booking:
        df_booking = pd.DataFrame(data_booking)
        df_booking = df_booking.iloc[:, 1:]
    else:
        print("No Past Booking data found in MongoDB!")
        df_booking = pd.DataFrame()

    return df_user, df_booking


def merge_data(df_user, df_booking):
    df_booking['User_ID'] = df_booking['User_ID'].astype(int)
    df = pd.merge(df_user, df_booking, on='User_ID', how="inner")
    return df


def convert_data_types(df):
    df['Total_Amount'] = df['Total_Amount'].astype(float)
    df['Discount_Applied'] = df['Discount_Applied'].astype(float)
    df['Tax_Amount'] = df['Tax_Amount'].astype(float)
    df['Final_Payment'] = df['Final_Payment'].astype(float)
    df['Location_ID'] = df['Service_Location'].astype(int)
    df['Service_ID'] = df['Service_ID'].astype(int)
    df['Provider_ID'] = df['Provider_ID'].astype(int)
    df["Booking_Date"] = pd.to_datetime(df["Booking_Date"], format="%d-%m-%Y")
    df["Service_Date"] = pd.to_datetime(df["Service_Date"], format="%d-%m-%Y")
    return df


def feature_engineering(df):
    df["Booking_hour"] = df["Booking_Date"].dt.hour
    df["Booking_day_of_week"] = df["Booking_Date"].dt.dayofweek
    df["Booking_Month"] = df["Booking_Date"].dt.month
    df["Service_hour"] = df["Service_Date"].dt.hour
    df["Service_day_of_week"] = df["Service_Date"].dt.dayofweek
    df["Service_Month"] = df["Service_Date"].dt.month
    df['service_lead_time'] = (df['Service_Date'] - df['Booking_Date']).dt.days
    return df


def map_time_slots(df):
    time_slot_mapping = {
        'Morning': (6, 12),
        'Afternoon': (12, 17),
        'Evening': (17, 22),
        'Night': (22, 6)
    }

    df['preferred_time_range'] = df['preferred_time_slots'].map(lambda slot: time_slot_mapping.get(slot, (0, 24)))
    df['preferred_time_start'] = df['preferred_time_range'].apply(lambda x: x[0])
    df['preferred_time_end'] = df['preferred_time_range'].apply(lambda x: x[1])
    df['time_slot_match'] = df.apply(lambda row: row['preferred_time_start'] <= row['Service_hour'] < row['preferred_time_end'], axis=1)
    df['time_slot_match'] = df['time_slot_match'].replace({'True': '1' ,'False': '0'})
    df.drop(columns=['preferred_time_start', 'preferred_time_end', 'preferred_time_range'], inplace=True)
    return df


def preprocess_dates(df):
    df['Booking_Date'] = pd.to_datetime(df['Booking_Date'])
    df['Service_Date'] = pd.to_datetime(df['Service_Date'])
    return df


def prepare_data_for_modeling(df):
    X = df[['gender', 'location_id','Service_ID','Provider_ID','preferred_services', 
            'preferred_time_slots', 'booking_history_count', 'cancellation_rate', 
            'average_rating_given', 'wallet_balance', 'subscription_status', 
            'Booking_Time', 'Service_Time','Booking_Month', 'Service_Month', 
            'service_lead_time']]

    y = df['Final_Payment']
    return X, y


def split_data(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Running the data processing pipeline

df_user, df_booking = load_data_from_mongo()
df = merge_data(df_user, df_booking)
df = convert_data_types(df)
df = feature_engineering(df)
df = map_time_slots(df)
df = preprocess_dates(df)
X, y = prepare_data_for_modeling(df)
X_train, X_test, y_train, y_test = split_data(X, y)

print(df.info())
# print(df['time_slot_match'].value_counts())

num_features = X.select_dtypes(include=['float64','int32','int64']).columns.tolist()
cat_features = X.select_dtypes(include=['bool','object']).columns.tolist()

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

if cat_features:
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
else:
    cat_transformer = 'passthrough'

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# print(df.info())
# a = df[['gender', 'location_id','Service_ID','Provider_ID','preferred_services', 
#             'preferred_time_slots', 'booking_history_count', 'cancellation_rate', 
#             'average_rating_given', 'wallet_balance', 'subscription_status', 
#             'Booking_Time', 'Service_Time','Booking_Month', 'Service_Month', 
#             'service_lead_time','Final_Payment']]

# sujal = a.select_dtypes(include=['float64','int32','int64'])
# corr_matrix = sujal.corr()

# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))

# # Create the heatmap
# sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap="coolwarm", linewidths=0.5)

# plt.show()

X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=num_features + cat_features)
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=num_features + cat_features)
joblib.dump(preprocessor, 'preprocessor.pkl')

print(X_train.shape)
print('preprocessor saved')

# # X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=num_features + cat_features)
# # X_test = pd.DataFrame(preprocessor.transform(X_test), columns=num_features + cat_features)


# # Feature Selection with XGBoost
# xgb_feat_selector = XGBRegressor(n_estimators=200, random_state=42)
# # xgb_feat_selector.fit(X_train, y_train)
# # selector = SelectFromModel(xgb_feat_selector, threshold="0.25*median", prefit=True)  # Retain more features

# # X_train = selector.transform(X_train)
# # X_test = selector.transform(X_test)
# # joblib.dump(selector, 'feature_selector.pkl')

# # print('Selector saved')

# # KFold Cross-Validation (Fix StratifiedKFold issue)
# # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # print("SUjal")
# # # Train Stacking Model with Diverse Base Models
# # stacking_model = StackingRegressor(
# #     estimators=[
# #         ('rf', RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42)),
# #         ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)),
# #         ('lgbm', LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=50, random_state=42))
# #     ],
# #     final_estimator=Ridge(alpha=1.0)
# # )

# # stacking_model.fit(X_train, y_train)

# # # Evaluate Model
# # y_pred_test = stacking_model.predict(X_test)
# # y_pred_train = stacking_model.predict(X_train)

# # print("Test Performance:", mean_absolute_error(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test)), r2_score(y_test, y_pred_test))
# # print("Train Performance:", mean_absolute_error(y_train, y_pred_train), np.sqrt(mean_squared_error(y_train, y_pred_train)), r2_score(y_train, y_pred_train))

# # joblib.dump(stacking_model, 'price_model.pkl')

if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)


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
study.optimize(objective, n_trials=10)

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
