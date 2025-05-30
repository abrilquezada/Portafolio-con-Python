# Portafolio-con-Python
Portafolios de inversión en Python basado en la Teoría Moderna de Portafolios:

## Descripción del proyecto

Este proyecto implementa y compara dos enfoques clásicos de teoría de portafolios: el portafolio mínima varianza (PMV) y el portafolio óptimo (M) basado en CAPM. Utilizando datos reales obtenidos desde Yahoo Finance, se realiza una optimización sujeta a restricciones prácticas (sin venta en corto y beta objetivo), calculando pesos óptimos, rendimientos esperados, riesgos y métricas de desempeño como el Sharpe Ratio.
1. Modelo de Markowitz (PMV) :
   - Riesgo-Rendimiento
   - Riesgo especifico
   - No considera un benchmark ni un 
activo libre de riesgo
2. Modelo de Sharpe (M): 
   - Riesgo-beta
   - Riesgo sistémico
   - Considera un benchmark para 
determinar al mercado y una tasa 
libre de riesgo para determinar la prima

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Requisitos](#requisitos)
3. [Estructura del Código](#estructura-del-código)
4. [Características Destacadas](#características-destacadas)
5. [Resultados y Comparaciones](#resultados-y-comparaciones)
6. [Conclusiones](#conclusiones)

## Requisitos 
```
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import plotly.express as px
from scipy.stats import norm
import random
```
