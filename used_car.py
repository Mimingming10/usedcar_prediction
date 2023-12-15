import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as py
import seaborn as sns

app_mode = st.sidebar.selectbox('Select Page', ['Home','Exploratory Data Analysis','Prediction'])
if app_mode == 'Home':
  st.title('USED CAR PRICE PREDICTION')
  st.image("mobil.jpg")

elif app_mode == 'Exploratory Data Analysis':

  # MEMANGGIL DATASET
  df = pd.read_csv('toyota.csv')
  st.header('Dataset : ')
  st.write(df)

  # JUMLAH MOBIL BERDASARKAN MODEL
  models = df.groupby('model').count()[['tax']].sort_values(by='tax', ascending=True).reset_index()
  models = models.rename(columns={'tax': 'numberOfCars'})

  fig = plt.figure(figsize=(15, 5))
  sns.barplot(x=models['model'],
              y=models['numberOfCars'],
              color='royalblue')
  plt.title("Jumlah Mobil Berdasarkan Model")
  plt.xticks(rotation=60)
  st.pyplot(plt.gcf())

  # UKURAN MESIN
  engine = df.groupby('engineSize').count()[['tax']].sort_values(by='tax').reset_index()
  engine = engine.rename(columns={'tax': 'count'})

  plt.figure(figsize=(15, 5))
  sns.barplot(x=engine['engineSize'], y=engine['count'], color='royalblue')
  plt.title("Ukuran Mesin Mobil")
  st.pyplot(plt.gcf())

  # DISTRIBUSI HARGA MOBIL
  plt.figure(figsize=(15, 5))
  sns.distplot(df['price'])
  plt.title("Distribusi Harga Mobil")
  st.pyplot(plt.gcf())

elif app_mode == 'Prediction':
  model = pickle.load(open('estimasi_mobil.sav', 'rb'))

  st.title("Estimasi Harga Mobil Bekas")

  # buat form dengan label "Input Data"
  with st.form("Input Data"):
    # buat input widget untuk setiap variabel
    year = st.number_input("Input Tahun Mobil")
    mileage = st.number_input("Input Km Mobil")
    tax = st.number_input("Input Pajak Mobil")
    mpg = st.number_input("Input Konsumsi BBM Mobil")
    engineSize = st.number_input("Input Engine Size")
    # buat tombol submit dengan label "Estimasi Harga"
    submitted = st.form_submit_button("Estimasi Harga")

  # jika tombol submit ditekan
  if submitted:
    # buat tuple dari input data
    input_data = (year, mileage, tax, mpg, engineSize)
    # prediksi harga mobil bekas dengan model
    predict = model.predict([[year, mileage, tax, mpg, engineSize]])
    # tampilkan hasil prediksi dalam EUR dan Rupiah
    st.write("Estimasi harga mobil bekas dalam EUR : ", predict)
    st.write("Estimasi harga mobil bekas dalam Rupiah : ", predict * 16000)
    # buat figure dan axes
    fig, ax = plt.subplots()
    # plot histogram untuk prediksi
    label = f"Input data: {input_data}"
    ax.hist(predict, bins=10, alpha=0.5, label=label)
    # tambahkan legend ke axes
    ax.legend()
    # tampilkan figure dengan st.pyplot
    st.pyplot(fig)


