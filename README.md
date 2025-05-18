This repository contains our solution for the M5 Forecasting - Accuracy competition on Kaggle. The goal of the challenge was to forecast 28 days of daily sales for Walmart across thousands of products and stores using historical data.

We explored multiple modeling strategies—ranging from tree-based models to recurrent neural networks—to evaluate trade-offs in performance, scalability, and resource eff

📌 Project Overview
Objective: Forecast daily sales for Walmart items over a 28-day horizon using historical data spanning multiple product categories, stores, and departments.

Approach:
We progressed through a range of modeling techniques:

* LightGBM for a fast and interpretable baseline

* LSTM and GRU for sequence modeling

* Seq2Seq GRU for one-shot multi-step forecasting

* Our final model, a GRU-based recurrent neural network, achieved the lowest error and best generalization on the Kaggle private leaderboard.

| Model       | Framework | Purpose                           | Key Highlights                                          |
| ----------- | --------- | --------------------------------- | ------------------------------------------------------- |
| LightGBM    | LightGBM  | Baseline with engineered features | Fast, interpretable; overfit slightly                   |
| LSTM        | Keras     | Time series with memory           | Better than baseline, but overfit                       |
| GRU         | Keras     | Lightweight time series model     | 🏆 Best generalization and performance                  |
| Seq2Seq GRU | Keras     | One-shot 28-day forecasting       | Fast and efficient, slightly underfit due to simplicity |

| Feature Type       | Description                        | Used In     | Purpose                         |
| ------------------ | ---------------------------------- | ----------- | ------------------------------- |
| Lag Features       | `sales_lag_7`, `sales_lag_28`      | LightGBM    | Weekly demand memory            |
| Rolling Stats      | `rolling_mean_7`, `rolling_std_28` | LightGBM    | Trend + volatility              |
| Price Signals      | `price_norm`, `price_momentum`     | LightGBM    | Price elasticity                |
| Time-Based         | `weekday`, `month`, `year`, etc.   | LightGBM    | Captures seasonality            |
| Event Flags        | Binary event day indicators        | LSTM/GRU    | Capture pre-holiday surges      |
| Sliding Sales Win. | Last 14 days raw sales per item    | LSTM/GRU    | Capture short-term patterns     |
| Transposed Matrix  | Items as columns, days as rows     | Seq2Seq GRU | Memory-efficient one-shot input |

| Submission File         | Model       | Private Score | Notes                                   |
| ----------------------- | ----------- | ------------- | --------------------------------------- |
| `submission.csv`        | LightGBM    | 5.39063       | High error, feature-rich baseline       |
| `submission_LSTM_*.csv` | LSTM        | 0.98608       | Strong model, slight overfit            |
| `submission_GRU_*.csv`  | GRU         | **0.78671**   | ✅ Best performance overall              |
| `simple_seq2seq_*.csv`  | Seq2Seq GRU | 0.99343       | Efficient but underfit without features |

** Key Learnings**
Tree-based models excel in interpretability and speed, especially with rich feature sets.

RNNs (especially GRUs) are better suited for capturing temporal dependencies without heavy feature engineering.

Simpler architectures like Seq2Seq GRU offer promising speed-performance trade-offs, especially in constrained environments.

Hyperparameter tuning was conducted manually to manage GPU memory limits and avoid OOM errors.


