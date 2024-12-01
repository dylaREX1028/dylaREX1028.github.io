
# Gabriel González Rivera - B93432
# Dilana Rodríguez Jiménez - C06660
# Sebastián Bonilla Vega - C01263

# Se importan las liberías necesarias

from fitter import Fitter
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Cargar los datos desde la base de datos
db_path = 'final.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM test_data", conn)
conn.close()

# Extraer la columna de interés
valores = df['variable_1']

# Ajustar distribuciones
f = Fitter(valores, distributions=['expon', 'gompertz', 'levy', 'logistic',
                                   'norm', 'rayleigh'])
f.fit()

# Mostrar la mejor distribución y sus parámetros
print("Mejor distribución encontrada:")
print(f.get_best(method='sumsquare_error'))


# Guardar el resumen gráfico para todos los datos
print("\nGuardando resumen gráfico para todos los datos...")
f.summary()
plt.savefig("resumen_grafico_todos_los_datos.png")
plt.close()

# Mejor distribución encontrada:
# {'logistic': {'loc': 0.9975756650579749, 'scale': 1.8860023725549602}}
# Fitted logistic distribution with error=0.012764

df_dia = df[df['sunlight'] > 0]  # Ajustar según tu definición de día
df_noche = df[df['sunlight'] == 0]  # Noche cuando no hay luz solar

# Ajuste de distribuciones para los datos del día
valores_dia = df_dia['variable_1']
f_dia = Fitter(valores_dia, distributions=['expon', 'gompertz', 'levy', 'logistic', 'norm', 'rayleigh'])
f_dia.fit()
print("\nMejor distribución encontrada para el día:")
print(f_dia.get_best(method='sumsquare_error'))

# Ajuste de distribuciones para los datos de la noche
valores_noche = df_noche['variable_1']
f_noche = Fitter(valores_noche, distributions=['expon', 'gompertz', 'levy', 'logistic', 'norm', 'rayleigh'])
f_noche.fit()
print("\nMejor distribución encontrada para la noche:")
print(f_noche.get_best(method='sumsquare_error'))


# Graficar y guardar el resumen gráfico para los datos del día
print("\nGuardando resumen gráfico para datos del día...")
f_dia.summary()
plt.savefig("resumen_grafico_dia.png")
plt.close()

# Graficar y guardar el resumen gráfico para los datos de la noche
print("\nGuardando resumen gráfico para datos de la noche...")
f_noche.summary()
plt.savefig("resumen_grafico_noche.png")
plt.close()