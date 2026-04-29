from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pickle import dump, load
import pandas as pd


def split_data(df: pd.DataFrame):
    y = df['selling_price']
    X = df[['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']]

    return X, y


def open_data(path="data/cars.csv"):
    df = pd.read_csv(path)
    df = df[['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    #преобразуем признаки из строковых в числовые
    cols = ['engine', 'mileage', 'max_power', 'torque']
    for col in cols:
        X_df[col] = pd.to_numeric(X_df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')
        X_df[col].fillna(X_df[col].median(), inplace=True)

    #объектные в числовые
    feats = ['owner', 'fuel', 'seller_type', 'transmission']
    X_df[feats] = X_df[feats].apply(lambda x: x.astype('category').cat.codes)
    
    # из названия авто извлекли бренд + сделали One-Hot Encoding для бренда 
    X_df['brand'] = X_df['name'].str.split().str[0]
    dummy = pd.get_dummies(X_df['brand'], prefix='brand')
    X_df = pd.concat([X_df, dummy], axis=1)
    X_df.drop(['brand', 'name'], axis=1, inplace=True)   

    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    MAE =mean_absolute_error(y_df, test_prediction)
    rmse = mean_squared_error(y_df, test_prediction) ** 0.5
    R2 = r2_score(y_df, test_prediction)    
    print(f"Model MAE is {MAE}, rsme is {rsme}, R2 is {R2}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    # encode_prediction_proba = {
    #     0: "Вам не повезло с вероятностью",
    #     1: "Вы выживете с вероятностью"
    # }

    # encode_prediction = {
    #     0: "Сожалеем, вам не повезло",
    #     1: "Ура! Вы будете жить"
    # }

    # prediction_data = {}
    # for key, value in encode_prediction_proba.items():
    #     prediction_data.update({value: prediction_proba[key]})

    # prediction_df = pd.DataFrame(prediction_data, index=[0])
    # prediction = encode_prediction[prediction]

    return prediction, prediction_proba   #prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
