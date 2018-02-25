# LSTM-on-Time-Series-Problem
## Practise Code
Hands-on time-series data (Regression problem). It includes some pre-processing.

## Data Source
Random Length Lumber Futures, Continuous Contract #1 (LB1) (Front Month)
###Description of Data
Historical Futures Prices: Random Length Lumber Futures, Continuous Contract #1. Non-adjusted price based on spot-month continuous contract calculations. Raw data from CME.
~~~
Link: https://www.quandl.com/data/CHRIS/CME_LB1-Random-Length-Lumber-Futures-Continuous-Contract-1-LB1-Front-Month?utm_medium=graph&utm_source=quandl
~~~

## Dependencies
~~~
python
tensorflow
keras
~~~

## Run
~~~
python lstm_time_series.py --data_path --look_back
~~~
