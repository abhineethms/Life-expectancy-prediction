import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    data = data.drop(['Country', 'Year'], axis=1)
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop('Life expectancy', axis=1)
    y = data['Life expectancy']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
