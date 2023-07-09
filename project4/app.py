from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from alpha_vantage.timeseries import TimeSeries

app = Flask(__name__)

# Menggunakan Alpha Vantage API
ALPHA_VANTAGE_API_KEY = '8CD9UTRCMSZXT59S'
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil simbol saham dari input pengguna
    stock_symbol = request.form['stock_symbol']

    try:
        # Mengambil data historis harga saham menggunakan Alpha Vantage API
        data, meta_data = ts.get_daily_adjusted(stock_symbol, outputsize='full')

        # Membuat DataFrame dari data historis harga saham
        df = pd.DataFrame(data).transpose()
        df.columns = ['open', 'high', 'low', 'close', 'volume', '1', '2', '3']

        # Mengubah tipe data kolom tanggal menjadi tipe datetime
        df.index = pd.to_datetime(df.index)

        # Memisahkan variabel input dan variabel target
        X = df.drop('close', axis=1)
        y = df['close']

        # Membangun model Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Menerima input tanggal pengguna melalui form HTML
        input_date = pd.to_datetime(request.form['input_date'])

        # Mengambil data input pengguna
        input_data = df.loc[input_date]

        # Memprediksi harga saham menggunakan input pengguna
        predicted_price = model.predict([input_data])[0]

        # Mengambil data harga saham untuk visualisasi grafik
        plot_data = df['close'].loc[:input_date]

        # Menghasilkan grafik harga saham
        plt.plot(plot_data.index, plot_data.values)
        plt.scatter(input_date, predicted_price, color='red', label='Prediksi')
        plt.title('Grafik Harga Saham')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga Saham')
        plt.legend()
        plot_path = 'static/plot.png'
        plt.savefig(plot_path)
        plt.close()

        return render_template('index.html', prediction=predicted_price, plot_path=plot_path)

    except Exception as e:
        error_message = "Error: " + str(e)
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
