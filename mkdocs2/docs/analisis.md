# Análisis de Resultados

Inicialmente, se realizaron gráficos utilizando los datos recopilados por 12 horas `(proyecto.db)`, logrando conseguir resultados satisfactorios. Esto se debe al comportamiento observado en las variables, del cual coincide con la teoría estudiada en clases, así validando de los resultados obtenidos. Respecto a la `variable_1`, los datos muestran una tendencia a aproximarse a la distribución en forma de campana de Gauss, o distribución normal. Este comportamiento demuestra cómo los valores tienden a concentrarse alrededor de un punto central, ya sea cercano a cero o al promedio. La agrupación de datos alrededor de este punto sugiere que la variable tienen características estables y predecibles. Además, la dispersión de los datos percibidos no es alta, lo que significa que los valores individuales no se alejan del promedio y la distribución observada es simétrica.

Por otro lado, la `variable_2` presenta un comportamiento diferente al anterior, mostrando una tendencia exponencial. Este es un fenómeno es habitual en eventos aleatorios. Un aspecto destacado de esta variable es su alto promedio en comparación con la `variable_1`, lo que indica la predominancia en su distribución. Sin embargo, esto viene de la mano con una alta dispersión, lo que sugiere una variabilidad significativa entre los datos individuales. Adicionalmente, la distribución de la `variable_2` es asimétrica, lo que indica un sesgo hacia un lado de la media, y presenta una kurtosis elevada. Este último punto representa la aparición de valores extremos, lo que implica un impacto considerable en el comportamiento global de los datos. 

Seguidamente, los resultados logrados con la base de datos recopilados por 24 horas `(final.db)`, se determina que el proceso aleatorio total, así como los datos separados en los grupos de día y noche, tienen como mejor ajuste una distribución logística. A partir de los parámetros `loc` y `scale` obtenidos mediante la herramienta fitter, se construyó una función de densidad de probabilidad que incorpora la variabilidad temporal del proceso. Al modelar con los parámetros se consigue una aproximación polinomial de grado 3, que permite evidenciar como evoluciona con el tiempo, y al normalizarla mejora la precisión del ajuste. Finalmente, al desnormalizar el tiempo, se consigue obtener una representación dinámica de la función de densidad de probabilidad demostrando las variaciones temporales del sistema.

Al evaluar los datos reunidos se comprueba que el proceso no es estacionario en sentido amplio. Esto se evidencia al percibir que la media de los valores conseguidos fluctúa notablemente a lo largo del tiempo, como se observa en la figura 8. En la variabilidad de la media se señala que no se cumple el criterio de constancia, esencial para la estacionaridad en sentido amplio. A pesar de identificar una tendencia a estabilizarse cerca de la medianoche, este no entra en el rango de lo aceptable para clasificar el proceso como estacionario. Igualmente, la frecuencia que se recopila podría influir en la interpretación de la variabilidad, sin embargo no se posee evidencia concluyente para afirmar este efecto.

Se analiza el promedio temporal de las funciones con la recopilación de los primeros valores de cada intervalo, consiguiendo un promedio temporal de aproximadamente 1.07. Este estudio ayuda a percibir la conducta de los datos, pero no es posible compararlo directamente con el promedio estadístico constante, debido a que este último  posee comportamiento fijo en el proceso. Este desempeño refleja una alta variabilidad en las características estadísticas del proceso, generando dificultad en la identificación.

En el estudio del proceso se obtiene un comportamieno no ergódico, dado que no cumple con los criterios de estacionaridad en sentido amplio. Al percibir la ausencia de un comportamiento constante en el promedio estadístico impide establecer una analogía con el promedio temporal. No obstante, se logra calcular el promedio temporal para las funciones muestra,  pero no puede utilizar para validar la ergodicidad debido a las fluctuaciones persistentes en los valores estadísticos del proceso. 

La relación entre el tiempo y la variable de interés muestra una correlación casi nula, lo que indica que no hay una conexión lineal significativa entre ambas; en otras palabras, los cambios en el tiempo no afectan directamente a los valores de la variable, lo que las hace prácticamente independientes. Sin embargo, la covarianza tiene un valor alto, lo que sugiere que ambas variables tienden a variar en la misma dirección, probablemente debido a patrones sistemáticos en los datos. Aunque no hay una proporción directa entre ellas, esta diferencia entre correlación y covarianza nos ayuda a entender cómo interactúan y se distribuyen las dependencias en el conjunto de datos.

La potencia promedio, calculada como el promedio de los valores al cuadrado de la señal, nos da una idea clara de cuánta energía contiene la señal a lo largo del tiempo. Con un valor cercano a 15, esta métrica resalta que la señal es estable y consistente en su intensidad. Además, resulta crucial para comparar la energía de la señal con el ruido, confirmando que la señal domina claramente, ya que el ruido tiene un impacto mínimo en la potencia total.

Por otro lado, la densidad espectral de potencia (PSD), obtenida mediante la Transformada Rápida de Fourier, muestra cómo se distribuye la energía de la señal entre las distintas frecuencias. La mayor parte de la energía se concentra en frecuencias bajas, con un pico importante cerca de la frecuencia cero, lo que sugiere que la señal varía lentamente y tiene un comportamiento suave y controlado. Aunque la PSD del ruido tiene un patrón similar, su magnitud es mucho menor, lo que refleja que el ruido está presente principalmente en frecuencias bajas, pero con un impacto limitado sobre la señal general.

El análisis del ruido confirma que su magnitud es significativamente menor comparada con la señal original. Esto se refuerza con la relación señal a ruido (SNR), que demuestra que el ruido es casi insignificante. Los gráficos de las señales original, ajustada y de ruido muestran cómo el filtrado aplicado reduce eficazmente las fluctuaciones no deseadas, conservando las características principales de la señal. Esto se observa especialmente en las etapas finales de los datos, donde las señales se estabilizan y el ruido disminuye notablemente. Aunque el ruido sigue un comportamiento similar al de la señal original, su magnitud es mucho más baja, lo que valida la efectividad del procesamiento aplicado.








