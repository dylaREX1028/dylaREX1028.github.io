# http://127.0.0.1:8000/ este va a ser el link del trabajo donde está el documento que corresponde
# al avance, de igual manera se va subir una archivo con el link.
# Gabriel González Rivera - B93432
# Dilana Rodríguez Jiménez - C06660
# Sebastián Bonilla Vega - C01263

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
from scipy.stats import skew, kurtosis
import numpy as np


# Conectar a la base de datos
conn = sqlite3.connect('proyecto.db')

# Consulta para extraer los datos de las dos variables
query = "SELECT variable_1, variable_2 FROM test_data"
data = pd.read_sql_query(query, conn)

# Cerrar la conexión
conn.close()

# Mostrar las primeras filas de los datos
print(data.head())

# Crear histogramas de variable_1 y variable_2
plt.figure(figsize=(12, 6))


# Histograma original de variable_1
plt.subplot(1, 2, 1)
plt.hist(data['variable_1'], bins=20, color='blue', alpha=0.7)
plt.title('Histograma de variable 1 (Original)')
plt.xlabel('variable 1')
plt.ylabel('Frecuencia')

# Histograma original de variable_2
plt.subplot(1, 2, 2)
plt.hist(data['variable_2'], bins=20, color='green', alpha=0.7)
plt.title('Histograma de variable 2 (Original)')
plt.xlabel('variable 2')
plt.ylabel('Frecuencia')

# Guardar la figura de los histogramas originales
plt.savefig('histogramas_originales.png')


# Crear una nueva figura para los histogramas con las líneas de mejor ajuste
plt.figure(figsize=(12, 6))

# Histograma con línea de mejor ajuste para variable_1
plt.subplot(1, 2, 1)
count1, bins1, ignored1 = plt.hist(data['variable_1'], bins=20,
                                   color='blue', alpha=0.7, density=False)
mu_1, std_1 = norm.fit(data['variable_1'])
best_fit_line1 = (norm.pdf(bins1, mu_1, std_1) *
                  count1.sum() * np.diff(bins1)[0])
plt.plot(bins1, best_fit_line1, 'k', linewidth=2)
plt.title('Histograma de variable 1 con ajuste gaussiano')
plt.xlabel('variable 1')
plt.ylabel('Frecuencia')

# Histograma de variable_2 con curva de ajuste
plt.subplot(1, 2, 2)
count2, bins2, ignored2 = (plt.hist(data['variable_2'], bins=20,
                                    color='green', alpha=0.7, density=False))
loc_2, scale_2 = expon.fit(data['variable_2'], floc=0)
# Fijar loc a 0 para mejorar el ajuste
best_fit_line2 = (expon.pdf(bins2, loc_2, scale_2)
                  * count2.sum() * np.diff(bins2)[0])
plt.plot(bins2, best_fit_line2, 'k', linewidth=2)
plt.title('Histograma de variable 2 con ajuste exponencial')
plt.xlabel('variable 2')
plt.ylabel('Frecuencia')
plt.savefig('histogramas_mejor_ajuste.png')

# Ajustar una distribución normal a variable_1
mu_1, std_1 = norm.fit(data['variable_1'])

# Graficar histograma y la curva ajustada
plt.figure(figsize=(6, 4))

# Cálculo de momentos de la variable_1
mean_1 = data['variable_1'].mean()
var_1 = data['variable_1'].var()
std_1 = data['variable_1'].std()
skew_1 = skew(data['variable_1'])
kurt_1 = kurtosis(data['variable_1'])

# Cálculo de momentos de la variable_2
mean_2 = data['variable_2'].mean()
var_2 = data['variable_2'].var()
std_2 = data['variable_2'].std()
skew_2 = skew(data['variable_2'])
kurt_2 = kurtosis(data['variable_2'])

# Mostrar resultados variable_1
print('Momentos de variable_1:')
print(f'Promedio: {mean_1}')
print(f'Varianza: {var_1}')
print(f'Desviación estándar: {std_1}')
print(f'Inclinación: {skew_1}')
print(f'Kurtosis: {kurt_1}')

# Mostrar resultados variable_2
print('Momentos de variable_2:')
print(f'Promedio: {mean_2}')
print(f'Varianza: {var_2}')
print(f'Desviación estándar: {std_2}')
print(f'Inclinación: {skew_2}')
print(f'Kurtosis: {kurt_2}')
