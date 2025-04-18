from flask import Flask, request, Response
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error  # ✅追加！
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import re

app = Flask(__name__)

@app.route('/')
def home():
    return '''
        <h2>株価予測フォーム</h2>
        <form method="POST" action="/predict">
            複数の銘柄コード（カンマ区切り、例: AAPL, MSFT）: <input type="text" name="tickers"><br>
            予測日（例: 2024-12-01）: <input type="text" name="date"><br>
            開始日（例: 2020-01-01）: <input type="text" name="start_date"><br>
            終了日（例: 2024-12-31）: <input type="text" name="end_date"><br>
            未来予測日数（例: 30日後）: <input type="text" name="forecast_days"><br>
            <input type="submit" value="予測する">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    tickers = request.form['tickers'].split(",")
    date_input = request.form['date']
    forecast_input = request.form['forecast_days']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    if '日後' in forecast_input:
        match = re.search(r'\d+', forecast_input)
        if match:
            days = int(match.group())
            date_obj = datetime.today() + timedelta(days=days)
            date_str = f"{days}日後（{date_obj.strftime('%Y-%m-%d')}）"
        else:
            return "日数の形式が正しくありません。例: '30日後'", 400
    elif date_input:
        date_obj = pd.to_datetime(date_input)
        date_str = date_input
    else:
        return "日付または予測日数を入力してください。", 400

    predictions = {}

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    result_html = "<h3>予測結果</h3>"

    for i, ticker in enumerate(tickers):
        ticker = ticker.strip()
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                predictions[ticker] = 'データなし'
                result_html += f'<h4>{ticker} のデータが取得できませんでした：データなし</h4>'
                continue
            data['Close'] = data['Close'].fillna(method='ffill')
        except Exception as e:
            predictions[ticker] = f'データ取得エラー: {str(e)}'
            result_html += f'<h4>{ticker} のデータが取得できませんでした：{str(e)}</h4>'
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        train_data = scaled_data[:int(len(scaled_data) * 0.8)]
        test_data = scaled_data[int(len(scaled_data) * 0.8):]

        def create_dataset(data, time_step=90):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 90
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        if X_train.size == 0 or X_test.size == 0:
            predictions[ticker] = 'データが少なすぎて予測できません'
            result_html += f'<h4>{ticker} のデータが少なすぎて予測できません</h4>'
            continue

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        predicted_price = predicted_stock_price[-1, 0]
        predictions[ticker] = predicted_price

        # ✅ MSE計算
        real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
        mse = mean_squared_error(real_stock_price, predicted_stock_price)

        # HTMLに予測結果とMSEを追加
        result_html += f'<h4>{ticker} の {date_str} の予測終値: {predicted_price:.2f} USD</h4>'
        result_html += f'<p>{ticker} のMSE（平均二乗誤差）: {mse:.2f}</p>'

        # グラフ描画
        color = colors[i % len(colors)]
        ax.plot(data.index, data['Close'], label=f'{ticker} 実株価', color=color, linewidth=2)
        test_dates = data.index[len(train_data):]
        ax.plot(test_dates[-len(predicted_stock_price):], predicted_stock_price, label=f'{ticker} 予測', color=color, linestyle='--', linewidth=2)

    ax.set_title('株価予測と実際の株価', fontsize=16, fontweight='bold', fontname='MS Gothic')
    ax.set_xlabel('日付', fontsize=12, fontname='MS Gothic')
    ax.set_ylabel('株価 (USD)', fontsize=12, fontname='MS Gothic')
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10, prop={'family': 'MS Gothic'})
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode('utf-8')

    result_html += f'<img src="data:image/png;base64,{graph_url}" />'

    return Response(result_html, content_type='text/html; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True)
    
