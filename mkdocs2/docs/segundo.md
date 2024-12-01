# Líneas de código importantes para final.db

Este primer código `(distribucion.py)` se centra en realizar un análisis del comportamiento de los datos en función del tiempo `(día y noche)`, identificando distribuciones estadísticas para cada conjunto de datos y presentando los resultados de forma óptima. Al cargar los datos desde una base de datos SQLite `(final.db)`, se trabaja con la columna `variable_1`.

Seguidamente se evalúan todas las características para conseguir la distribución que mejor se ajusta, este modelo utiliza el método de error cuadrático medio `(sumsquare_error)` y grafican los resultados. Finalmente, se dividen en día y noche, con base a los valores de una columna `sunlight`. Para cada grupo, se repite el ajuste a las mismas distribuciones y se guarda los gráficos.
  

```python
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
```


En el caso del segundo código `avance_final.py` de igual manera se almacena los valores en la base de datos y se procesan para calcular el tiempo relativo a la medianoche en minutos. Para esto se ajusta los valores, generando que los datos posteriores al mediodía sean negativos y graficándolos. Adicionalmente, el código analiza los datos en intervalos específicos, ajustando distribuciones logísticas que modelan su conducta. Los parámetros clave de estas distribuciones, como la media `(loc)` y la escala `(scale)`, se estudian a lo largo del tiempo, mostrando su evolución mediante gráficos. 

Para entender mejor la relación temporal, se realiza un ajuste polinomial de grado 3 sobre estos parámetros, normalizando primero los datos temporales para mejorar el ajuste. Después, estos polinomios normalizados se desnormalizan para interpretar los resultados en el contexto original.

Además, en el código se demuestra la señal en términos de frecuencia aplicando la Transformada Rápida de Fourier (FFT), así separando la señal útil del ruido por medio de rangos de frecuencia definidos. Calcula y evalua las métricas clave como la densidad espectral de potencia (PSD), la relación señal a ruido (SNR), estadísticas como la desviación estándar, varianza, o relación pico a pico y realiza una reconstrucción de la señal ajustada. 

 



```python
import pandas as pd
import sqlite3
import numpy as np
from fitter import Fitter
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Se cargan los datos desde la base de datos
db_path = 'final.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM test_data", conn)
conn.close()


# Se realiza primero el análisis durante todo el día

# Se convierte la columna de tiempo a datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Se acomoda el tiempo relativo a las 12 am
df['minutes_since_midnight'] = (
    df['timestamp'].apply(lambda x:
                          (x - x.replace(hour=0, minute=0, second=0,
                                         microsecond=0)).total_seconds() / 60))

df['minutes_since_midnight'] = (
    df['minutes_since_midnight'].apply(lambda x:
                                       x - 1440 if x > 720 else x))

# Se grafica la totalidad de los datos con el tiempo relativo a las 12 am.
plt.figure(figsize=(12, 6))
plt.plot(df['minutes_since_midnight'], df['variable_1'], 'o-', markersize=3,
         label='Datos Recopilados')
plt.axvline(x=0, color='red', linestyle='--', label='12:00 AM')
plt.xlabel('Tiempo relativo a las 12:00 AM (minutos)')
plt.ylabel('Valores de variable 1')
plt.title('Datos recopilados con respecto al tiempo relativo a las 12:00 AM')
plt.legend()
plt.grid(True)
plt.savefig('Gráfico_de_datos_según_el_tiempo.png')


# Se agrupan los datos en intervalos de 5 min y se calcula el promedio
df_grouped = (
    df.groupby(df['minutes_since_midnight'] // 5 * 5)['variable_1'].mean())

# Se grafica el promedio por intervalos de 5 minutos
plt.figure(figsize=(12, 6))
plt.plot(df_grouped.index, df_grouped.values, '-o', markersize=4,
         label='Promedio por Intervalo (5 minutos)')
plt.axvline(x=0, color='red', linestyle='--', label='12:00 AM')
plt.xlabel('Tiempo Relativo a las 12:00 AM (minutos)')
plt.ylabel('Promedio de variable 1')
plt.title('Promedio de datos recopilados por intervalos de 5 minutos')
plt.legend()
plt.grid(True)
plt.savefig('Gráfico_promediado_datos_según_el_tiempo.png')


# Para los parámetros
# se agrupan por cada 5 min
df['time_bin'] = (df['minutes_since_midnight'] // 5) * 5

# Inicializar listas para almacenar parámetros ajustados
loc_values = []
scale_values = []
time_points = []

# se ajusta una distribución logística para cada intervalo de 5 minutos
for time_bin, group in df.groupby('time_bin'):
    valores = group['variable_1']
    f = Fitter(valores, distributions=['logistic'])
    f.fit()

    # Se obtienen los mejores parámetros para la distribución logística
    params = f.get_best(method='sumsquare_error')['logistic']

    # se almacenan los parámetros y tiempo correspondiente
    loc_values.append(params['loc'])
    scale_values.append(params['scale'])
    time_points.append(time_bin)

# Se grafica la evolución de los parámetros con el tiempo
plt.figure(figsize=(12, 6))
plt.plot(time_points, loc_values, label='loc (μ)', marker='o')
plt.plot(time_points, scale_values, label='scale (s)', marker='o')
plt.xlabel('Tiempo en Minutos Relativo a las 12:00 AM')
plt.ylabel('Valor del Parámetro')
plt.title('Evolución temporal de (μ) y (s) en la distribución logística')
plt.legend()
plt.grid(True)

plt.savefig('Gráfico_evolución_de_parámetros.png')

# Se convierten los time_points a un arreglo de NumPy
time_points_np = np.array(time_points)

# se hace el ajuste polinomial de grado 2 o 3
poly_loc = Polynomial.fit(time_points_np, loc_values, deg=3)
poly_scale = Polynomial.fit(time_points_np, scale_values, deg=3)

# Se generan los puntos para graficar las aproximaciones
time_fine = np.linspace(time_points_np.min(), time_points_np.max(), 1000)

# Se grafica el ajuste polinomial
plt.figure(figsize=(12, 6))
plt.plot(time_points_np, loc_values, 'o-', label='loc (Datos)', markersize=4)
plt.plot(time_fine, poly_loc(time_fine), label='loc (Aprox. Polinomial)',
         linestyle='--', linewidth=2)
plt.plot(time_points_np, scale_values, 'o-', label='scale (Datos)',
         markersize=4)
plt.plot(time_fine, poly_scale(time_fine), label='scale (Aprox. Polinomial)',
         linestyle='--', linewidth=2)
plt.xlabel('Tiempo en Minutos Relativo a las 12:00 AM')
plt.ylabel('Valor del Parámetro')
plt.title('Aproximación Polinomial de loc(t) y scale(t)')
plt.legend()
plt.grid(True)

plt.savefig('Gráfico_aprox_polinomial_grado3.png')


# Se Normaliza el tiempo para mejorar el ajuste
time_points_normalized = ((time_points_np - time_points_np.mean()) /
                          time_points_np.std())

# Se ajustan los polinomios con tiempo normalizado
poly_loc = Polynomial.fit(time_points_normalized, loc_values, deg=3)
poly_scale = Polynomial.fit(time_points_normalized, scale_values, deg=3)

# Se convierten a la forma clásica
poly_loc_standard = poly_loc.convert()
poly_scale_standard = poly_scale.convert()

# Se muestran las nuevas expresiones polinomiales
print("Expresión polinomial para loc (μ) con tiempo normalizado:")
print(poly_loc_standard)

print("\nExpresión polinomial para scale (s) con tiempo normalizado:")
print(poly_scale_standard)


# Se obtienen los coeficientes de los polinomios normalizados
loc_coeffs = poly_loc.convert().coef
scale_coeffs = poly_scale.convert().coef

# Se obtiene la media y desviación estándar del tiempo original
mu_t = time_points_np.mean()  # Media del tiempo original
sigma_t = time_points_np.std()  # Desviación estándar del tiempo original


# Se hace una función para desnormalizar
def desnormalizar_polinomio(coeffs, mu_t, sigma_t):
    """
    Desnormaliza un polinomio ajustado en términos de tiempo normalizado
    para obtenerlo en términos de tiempo original.
    """
    degree = len(coeffs) - 1
    desnormalizado = Polynomial([0])  # Polinomio inicial vacío

    for i in range(degree + 1):
        if i == 0:
            # Para el término constante, no hay ajuste
            desnormalizado += Polynomial([coeffs[i]])
        else:
            # Ajuste para los términos no constantes
            desnormalizado += (
                Polynomial.fromroots([mu_t / sigma_t] * i).convert() *
                coeffs[i] / (sigma_t ** i))

    return desnormalizado.convert()


# Se desnormalizan los polinomios
poly_loc_desnormalizado = desnormalizar_polinomio(loc_coeffs, mu_t, sigma_t)
poly_scale_desnormalizado = desnormalizar_polinomio(scale_coeffs, mu_t,
                                                    sigma_t)

# Se muestran las expresiones desnormalizadas
print("Expresión polinomial desnormalizada para loc (μ):")
print(poly_loc_desnormalizado)

print("\nExpresión polinomial desnormalizada para scale (s):")
print(poly_scale_desnormalizado)


# Para los Promedios temporales de funciones de muestra

# Se agrega el índice de recopilación
df['collection_index'] = df.index // 100  # Cada 100 filas representa una
# recopilación de datos

# Se filtran los primeros  valores de cada recopilación
primeros_valores = df[df.index % 100 == 0]


# Se calcula el promedio temporal
promedio_temporal_primeros = primeros_valores['variable_1'].mean()
print(
    f"Promedio temporal de los primeros valores de cada recopilación: "
    f"{promedio_temporal_primeros}"
)

# Se grafican los valores recopilados y el
plt.figure(figsize=(12, 6))
plt.plot(primeros_valores['minutes_since_midnight'],
         primeros_valores['variable_1'], '-o',
         label='Primeros Valores de Cada Recopilación', markersize=4)
plt.axhline(y=promedio_temporal_primeros,
            color='red', linestyle='--', linewidth=2,
            label=f'Promedio Temporal = {promedio_temporal_primeros:.2f}')

plt.axvline(x=0, color='blue', linestyle='--', linewidth=1, label='12:00 AM')
plt.xlabel('Tiempo Relativo a las 12:00 AM (minutos)')
plt.ylabel('Valores de variable 1')
plt.title('Primeros valores de cada recopilación y su promedio temporal')
plt.legend()
plt.grid(True)
plt.savefig('Gráfico_promedio_funcion_muestra.png')

# Se calculan la correlacion y la covarianza
covarianza = df['minutes_since_midnight'].cov(df['variable_1'])  # Covarianza
correlacion = df['minutes_since_midnight'].corr(df['variable_1'])  # Correlación

print("Covarianza entre el Tiempo y la Variable:", covarianza)
print("Correlación Tiempo y la Variable:", correlacion)


# Se calcula el cuadrado de la variable_1 (X^2(t))
df['variable_1_squared'] = df['variable_1'] ** 2
# Número de puntos de la señal
N = len(df)
print("Numero de datos:", N)

# Paso 1: Filtrar registros con tiempos únicos
df['time_diff'] = df['minutes_since_midnight'].diff()

# Filtrar donde la diferencia de tiempo sea mayor que 0
df_filtered = df[df['time_diff'] > 0]

# Eliminar la columna auxiliar 'time_diff'
df_filtered = df_filtered.drop(columns='time_diff')

# Paso 2: Calcular el intervalo de tiempo (dT) en segundos
dT = (df_filtered['minutes_since_midnight'].iloc[1] - df_filtered['minutes_since_midnight'].iloc[0])   # Intervalo en segundos

# Paso 3: Extraer la variable de interés (por ejemplo 'variable_1')
variable = df_filtered['variable_1'].values

# Paso 4: Realizar la transformada rápida de Fourier (FFT)
fft_result = np.fft.fft(variable)

# Paso 5: Calcular las frecuencias correspondientes utilizando np.fft.fftfreq
n = len(df_filtered)  # Número de muestras
frequencies = np.fft.fftfreq(n, d=dT)

# Paso 6: Calcular la densidad espectral de potencia (PSD)
psd = np.abs(fft_result)**2 / (n * dT)

# Paso 7: Graficar la densidad espectral de potencia
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:n//2], psd[:n//2])  # Solo mostrar frecuencias positivas
plt.title('Densidad Espectral de Potencia (PSD)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia')
plt.grid(True)
plt.show()

# Definir un rango de frecuencias bajas para la señal (puedes ajustar el rango)
low_freq_range = (frequencies > 0) & (frequencies < 0.5)  # ejemplo: frecuencias menores a 0.1 Hz

# Definir un rango de frecuencias altas para el ruido (ajusta el rango según sea necesario)
high_freq_range = (frequencies > 0.5) & (frequencies < 10)  # ejemplo: frecuencias entre 0.1 y 10 Hz

# Potencia de la señal: solo la parte baja de las frecuencias
signal_power = np.sum(psd[low_freq_range])

# Potencia del ruido: solo la parte alta de las frecuencias
noise_power = np.sum(psd[high_freq_range])
# Agregar un pequeño valor al ruido si es cero para evitar la división por cero
if noise_power == 0:
    noise_power = 1e-10  # Un valor pequeño para evitar la división por cero

# Calcular SNR
snr = signal_power / noise_power
print(f"SNR (relación señal a ruido): {snr}")

# Crear una copia de fft_result para filtrar solo las frecuencias bajas
fft_signal = np.zeros_like(fft_result)
fft_signal[low_freq_range] = fft_result[low_freq_range]

# Reconstruir la señal ajustada a partir de las frecuencias bajas
adjusted_signal = np.fft.ifft(fft_signal).real

# Calcular el ruido: diferencia entre la señal original y la ajustada
noise = df_filtered['variable_1'] - adjusted_signal
noise_std = noise.std()
print(f"Desviación estándar del ruido: {noise_std}")

noise_fft = np.fft.fft(noise)
noise_psd = np.abs(noise_fft)**2 / len(noise)
plt.plot(frequencies, noise_psd, label="PSD del ruido")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia")
plt.legend()
plt.show()
correlation = np.corrcoef(df_filtered['variable_1'], noise)[0, 1]
print(f"Coeficiente de correlación entre señal y ruido: {correlation}")





plt.figure(figsize=(12, 6))

# Graficar todas las curvas juntas
plt.plot(df_filtered.index, df_filtered['variable_1'], label='Señal original', color='blue', alpha=0.7)
plt.plot(df_filtered.index, adjusted_signal, label='Señal ajustada', color='green', linestyle='--')
plt.plot(df_filtered.index, noise, label='Ruido', color='red', linestyle=':')

# Configuración del gráfico
plt.title('Señal Original, Ajustada y Ruido')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Potencia promedio (promedio de X^2(t))
potencia_promedio = df['variable_1_squared'].mean()

print(f"Potencia Promedio: {potencia_promedio}")

spectral_flatness = np.exp(np.mean(np.log(noise_psd + 1e-10))) / np.mean(noise_psd + 1e-10)
print(f"Planitud espectral del ruido: {spectral_flatness}")

# Relación pico a pico de la señal ajustada
signal_peak_to_peak = adjusted_signal.max() - adjusted_signal.min()

# Relación pico a pico del ruido
noise_peak_to_peak = noise.max() - noise.min()

print(f"Relación pico a pico de la señal ajustada: {signal_peak_to_peak}")
print(f"Relación pico a pico del ruido: {noise_peak_to_peak}")

# Varianza de la señal ajustada
signal_variance = np.var(adjusted_signal)
signal_std_dev = np.std(adjusted_signal)

# Varianza del ruido
noise_variance = np.var(noise)
noise_std_dev = np.std(noise)

print(f"Varianza de la señal ajustada: {signal_variance}")
print(f"Desviación estándar de la señal ajustada: {signal_std_dev}")
print(f"Varianza del ruido: {noise_variance}")
print(f"Desviación estándar del ruido: {noise_std_dev}")
```



En el caso de la base de datos, se pueden descaragar en el siguiente link:
- [Descargar base de datos](archivos/final.db)









