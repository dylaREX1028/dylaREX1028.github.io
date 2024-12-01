# Conclusiones

**1.**  En síntesis, los gráficos generados a partir de los datos recopilados durante 12 horas (`proyecto.db`) demuestran un comportamiento equivalente con la teoría estudiada, así logrando validar los resultados conseguiodos. En el caso de la `variable_1` muestra una conducta de una distribución normal, caracterizada por la concentración de valores cerca de un punto, con baja dispersión y una distribución simétrica, este comportamiento señala la estabilidad y predictibilidad en los datos. En contraste, con la `variable_2` posee una distribución exponencial, con un promedio más alto que la `variable_1`, alta dispersión, asimetría y kurtosis elevada. Esta conduca indica la presencia de valores extremos que influyen en el comportamiento global de esta variable.

**2.**  En el análisis de los datos conseguidos durante 24 horas (`final.db`) se observa que el proceso, tanto en conjunto como separado en los grupos de día y noche, se logra un ajuste a una distribución logística. Además, con la ayuda de la herramienta `fitter` se ajusta y construye una función de densidad de probabilidad que pueda incorporar la variabilidad temporal del proceso. Este modelo temporal utiliza una aproximación polinomial de grado 3 facilitando la observación de patrones dinámicos e incrementando la precisión del ajuste al normalizar los datos.

**3.**  Se evidencia que el proceso no es estacionario en sentido amplio, dado que la media varia con el tiempo. Aunque se observa una tendencia a estabilizarse cerca de la medianoche, esta no cumple con los criterios para clasificar al proceso como estacionario. Por otro lado, en el cálculo del promedio temporal, se obtiene un resulado de 1.07, siendo este útil para observar comportamientos generales en el sistema, sin embargo, no se puede comparar con el promedio estadístico debido a la naturaleza no estacionaria del proceso.

**4.**  Se concluye que el proceso no es ergódico, ya que no cumple con los criterios para establecer equivalencias entre el promedio temporal y el promedio estadístico. Debido a las fluctuaciones ocurrentes en los valores estadísticos imposibilitan la valición.

**5.**  La relación entre el tiempo y la variable no es lineal, ya que la correlación es casi nula. Sin embargo, la covarianza alta refleja una tendencia conjunta en su variación, probablemente causada por patrones sistemáticos en los datos. Esto indica que, aunque no están directamente relacionadas, existen dependencias más complejas que influyen en su comportamiento.

**6.**  La potencia promedio, con un valor estable y consistente, confirma que la señal es predominante frente al ruido. Esto asegura que la energía de la señal es suficientemente alta como para no verse afectada significativamente por el ruido, validando su estabilidad y utilidad en aplicaciones posteriores.

**7.**  La PSD revela que la energía de la señal se concentra en frecuencias bajas, lo que refleja un comportamiento suave y controlado. Aunque el ruido comparte este patrón, su magnitud es mucho menor, lo que garantiza que el impacto del ruido sobre la señal es mínimo.

**8.**  El ruido es insignificante en comparación con la señal, como lo demuestra la alta relación señal a ruido (SNR). Además, el filtrado aplicado elimina eficazmente las fluctuaciones no deseadas, permitiendo que la señal conserve sus características esenciales. En las etapas finales del análisis, las señales se estabilizan, lo que refuerza la efectividad del procesamiento realizado.


