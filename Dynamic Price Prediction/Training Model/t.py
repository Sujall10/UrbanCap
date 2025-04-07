from pymongo import MongoClient
import pandas as pd
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

# Load Data (MongoDB or Local CSV)
def load_data():
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

def preprocess_data(df_user, df_booking):
    df_booking['User_ID'] = df_booking['User_ID'].astype(int)
    df = pd.merge(df_user, df_booking, on='User_ID', how='inner')
    df['Booking_Date'] = pd.to_datetime(df['Booking_Date'], format='%d-%m-%Y')
    df['Service_Date'] = pd.to_datetime(df['Service_Date'], format='%d-%m-%Y')
    df['service_lead_time'] = (df['Service_Date'] - df['Booking_Date']).dt.days
    df['Booking_hour'] = df['Booking_Date'].dt.hour
    df['Booking_day_of_week'] = df['Booking_Date'].dt.dayofweek
    df['Booking_Month'] = df['Booking_Date'].dt.month
    df['Service_hour'] = df['Service_Date'].dt.hour
    df['Service_day_of_week'] = df['Service_Date'].dt.dayofweek
    df['Service_Month'] = df['Service_Date'].dt.month
    return df

def prepare_data(df):
    features = ['gender', 'location_id', 'Service_ID', 'Provider_ID', 'preferred_services',
                'preferred_time_slots', 'booking_history_count', 'cancellation_rate',
                'average_rating_given', 'wallet_balance', 'subscription_status',
                'Booking_hour', 'Service_hour', 'Booking_Month', 'Service_Month',
                'service_lead_time']
    target = 'Final_Payment'
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline(X_train):
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]) if cat_features else 'passthrough'
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    return preprocessor

def objective(trial, X_train, y_train, preprocessor):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = XGBRegressor(**params, random_state=42, eval_metric='rmse')
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='neg_root_mean_squared_error').mean()
    return -score

def train_model(X_train, y_train, preprocessor):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, preprocessor), n_trials=10)
    best_params = study.best_params
    print("Best Parameters:", best_params)
    final_model = XGBRegressor(**best_params, random_state=42, eval_metric='rmse')
    final_pipeline = Pipeline([('preprocessor', preprocessor), ('model', final_model)])
    final_pipeline.fit(X_train, y_train)
    joblib.dump(final_pipeline, 'Final_Model.pkl')
    return final_pipeline

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("\nTraining Performance:")
    print(f"R² Score: {r2_score(y_train, y_train_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print("\nTest Performance:")
    print(f"R² Score: {r2_score(y_test, y_test_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

def main():
    df_user, df_booking = load_data()
    df = preprocess_data(df_user, df_booking)
    X_train, X_test, y_train, y_test = prepare_data(df)
    preprocessor = build_pipeline(X_train)
    model = train_model(X_train, y_train, preprocessor)
    evaluate_model(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
