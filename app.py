import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page(): 
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    # image = Image.open('data/cars.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Cars",
        # page_icon=image,

    )

    st.write(
        """
        # Машины
        Оценка стоимости машин.
        """
    )

    # st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction):
    st.write("## Предсказание")
    st.write(prediction)

    # st.write("## Вероятность предсказания")
    # st.write("prediction_probas")


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction= load_model_and_predict(user_X_df)
    write_prediction(prediction)


def sidebar_input_features():
    # 'name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats'
    name = st.text_input("Введите марку и модель машины", placeholder="e.g. Skoda Rapid 1.5 TDI Ambition")
    year = st.sidebar.slider("Год выпуска", min_value=1885, max_value=2026, value=2026, step=1)
    km_driven = st.sidebar.slider("Пробег", min_value=0, max_value=1000000, value=0, step=1000)
    fuel = st.sidebar.selectbox("Топливо", ('Diesel', 'Petrol', 'LPG', 'CNG'))
    seller_type = st.sidebar.selectbox("Продавец", ('Individual', 'Dealer', 'Trustmark Dealer'))
    transmission = st.sidebar.selectbox("Трансмиссия", ('Manual', 'Automatic'))
    owner = st.sidebar.selectbox("Владелец", ('First Owner',  'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'))
    mileage = st.slider("Введите расход топлива", min_value=0, max_value=50, value=10, step=0.1)    
    engine = st.slider("Введите двигатель", min_value=0, max_value=5000, value=1500, step=1)    
    max_power = st.slider("Введите мощность", min_value=0, max_value=150, value=90, step=1)    
    torque = st.slider("Введите крутящий момент двигателя", , min_value=0, max_value=500, value=200, step=1)  
    seats = st.sidebar.slider("Число сидений", min_value=1, max_value=100, value=5, step=1)
       
    data = {
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
