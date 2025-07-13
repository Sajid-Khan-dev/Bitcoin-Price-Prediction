## 📄 Project Description
Bitcoin Price Prediction using CatBoost & Optuna
This project predicts Bitcoin (BTC-USD) closing prices using real historical data and technical indicators. It uses CatBoostRegressor for machine learning and Optuna to optimize the model's hyperparameters. The model is evaluated using RMSE and visualized against actual prices.


## 🔍 How It Works – Step-by-Step

1. **📥 Load Bitcoin Data**  
   - Uses `yfinance` to fetch BTC-USD price data from 2020 to 2024.
   - Extracts OHLCV (Open, High, Low, Close, Volume) values.

2. **📊 Generate Technical Indicators**  
   - Calculates features using `ta` (Technical Analysis) library:
     - Moving Averages (MA5 to MA200)
     - RSI (Relative Strength Index)
     - Stochastic RSI (%K, %D)
     - MACD & Signal Line
     - Bollinger Bands (mid, high, low)

3. **🧹 Prepare Dataset**  
   - Combines OHLCV + indicators into a single DataFrame.
   - Target variable: `Close` price.
   - Splits data into training (80%) and testing (20%).

4. **🤖 Train CatBoost Model**  
   - Trains a `CatBoostRegressor`, which is efficient and handles numerical data well.

5. **🎯 Optimize with Optuna**  
   - Defines an objective function to:
     - Suggest random parameters (`learning_rate`, `depth`, etc.)
     - Train and evaluate model
     - Return RMSE to minimize
   - Runs at least 25 trials to find the best hyperparameters(I used 5 trials).

6. **📉 Final Training & Prediction**  
   - Retrains model using the best parameters found.
   - Makes predictions on test data.
   - Calculates **RMSE** (Root Mean Squared Error).

7. **📈 Visualization**  
   - Plots actual vs predicted Bitcoin prices using `matplotlib`.

---

## 📊 RMSE (Evaluation Metric)

RMSE measures how far predictions are from actual values.  
Lower RMSE means better predictions.

```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
