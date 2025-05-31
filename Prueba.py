
#Optimización de portafolios de 10 acciones

#Instalar las paqueterías adecuadas para el funcionamiento correcto del código
from turtle import reset
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import plotly.express as px
from scipy.stats import norm
import random
from scipy.optimize import minimize

#Nota: Para instalar "quantstats" fue necesario instalar la última versión

#Selección de las 10 acciones para poder desarrollar los portafolios

tickers = ['MSFT', 'IBM', 'MCD', 'V', 'AXP', 'HD', 'WMT', 'CAT', 'KO', 'AAPL']
start = '2024-01-01'
data = yf.download(tickers, start=start)
data_benchmark = yf.download('^GSPC', start=start)
print(data.head())
print(data_benchmark.head())

#Obtenemos los rendimientos diarios de cada acción así como los rednimientos diarios
#del benckmark
returns_df = data['Close'].pct_change(fill_method=None).dropna()
benchmark_returns_df = data_benchmark['Close'].pct_change(fill_method=None).dropna()

#Obtenemos las métricas clave: rendimiento esperado y matriz de covarianza
avg_returns = returns_df.mean() * 252
cov_mat = returns_df.cov() * 252

#Para el portafolio Óptimo (M)

num_portfolios = 1500
n_assets = len(tickers)

weights = np.random.random(size=(num_portfolios, n_assets))
weights /= np.sum(weights, axis=1)[:, np.newaxis]

portf_rtns = np.dot(weights, avg_returns)
portf_rtns

portf_vol = []
for i in range(0, len(weights)):
 portf_vol.append(np.sqrt(np.dot(weights[i].T,np.dot(cov_mat, weights[i]))))

portf_vol = np.array(portf_vol)

portf_sharpe_ratio = portf_rtns / portf_vol

portf_results_df = pd.DataFrame({
    'returns': portf_rtns,
    'volatility':  portf_vol,
    'sharpe_ratio': portf_sharpe_ratio
})

print(portf_results_df)

max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

print('Maximum sharpe ratio portfolio')
print('Permformance:')
for index, value in max_sharpe_portf.items():
  if index != 'sharpe_ratio':
      print(f'\n{index}: {100 * value:.2f}% ', end="")
  else:
      print(f'\n{index}: {value:.2f}', end="")
print('\n \nWeights:')

for x, y in zip(tickers, weights[np.argmax(portf_results_df.sharpe_ratio)]):
  print(f'{x}: {100 * y:.2f} %', end="")

#Para la beta

returns_df = returns_df.loc[benchmark_returns_df.index]
Rf = 0.05 / 252

betas = {}
for col in returns_df.columns:
  cov = returns_df[col].cov(benchmark_returns_df['^GSPC'])
  var = benchmark_returns_df['^GSPC'].var()
  calcular_beta = cov / var
  betas[col] = calcular_beta

Rm = benchmark_returns_df['^GSPC'].mean() *252

weights =  {
    'MSFT': 0.0792,
    'IBM': 0.0426,
    'MCD': 0.1961,
    'V': 0.0242,
    'AXP': 0.1429,
    'HD': 0.0751,
    'WMT': 0.0201,
    'CAT': 0.1153,
    'KO': 0.0221,
    'AAPL': 0.2824
}

betas_series = pd.Series(betas)
weights_series = pd.Series(weights)

portfolio_beta = (betas_series * weights_series).sum()

print(f"Beta del portafolio: {portfolio_beta:.4f}")

#Para el CAPM 

CAPM = Rf + portfolio_beta * (Rm - Rf)
print(CAPM)


#Para el portafolio mínima varianza (PMV)
# Función para calcular el riesgo (varianza del portafolio)


def portfolio_risk(weights, cov_mat):
    return np.dot(weights.T, np.dot(cov_mat, weights))

# Función objetivo para minimizar 
def objective_function(weights, avg_returns, cov_mat, risk_aversion=1):
    portfolio_return = np.dot(weights, avg_returns)
    portfolio_risk_value = portfolio_risk(weights, cov_mat)
    return risk_aversion * portfolio_risk_value - portfolio_return

# Restricciones: suma de pesos igual a 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Límites: pesos entre 0 y 1 
bounds = [(0, 1) for _ in range(len(avg_returns))]


# Cálculo de la varianza con los pesos óptimos
result = minimize(
    portfolio_risk,
    list(weights.values()), 
    args=(cov_mat,),
    method='SLSQP',
    bounds=bounds,
    constraints= constraints
)

optimal_weights = result.x

print(optimal_weights) 

# Cálculo de la varianza con los pesos óptimos

 
varianza_portafolio = portfolio_risk(optimal_weights, cov_mat)
print("Varianza del portafolio óptimo:", varianza_portafolio)

# Cálculo del retorno esperado con los pesos óptimos
retorno_esperado = np.dot(optimal_weights, avg_returns)*100
print("Retorno esperado del portafolio óptimo:", retorno_esperado)

for i, w in enumerate(optimal_weights*100):
    print(f"Activo {i+1}: {w:.2f}%")

riesgo_varianza = np.dot(optimal_weights.T, np.dot(cov_mat, optimal_weights))
print("Varianza del portafolio:", riesgo_varianza)
riesgo_std = np.sqrt(riesgo_varianza)*100
print("Riesgo:", riesgo_std)

Rf = 0.05

# Índice de Sharpe
sharpe_ratio = ( retorno_esperado - Rf) / riesgo_std

print("Índice de Sharpe:", sharpe_ratio)

#Beta del portafolio

beta_portafolio = np.dot(optimal_weights, betas_series)

print("Beta del portafolio:", beta_portafolio)
