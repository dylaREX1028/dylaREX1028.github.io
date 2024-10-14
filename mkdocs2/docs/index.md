# Introducción


En el presente proyecto tiene la finalidad de analizar un conjunto de datos utilizando el lenguaje de programación Python. Los datos se recopilan de una base de datos SQLite, se procesan y se utilizan técnicas estadísticas para ajustarlas a los modelos según lo estudiado en clase.

## Código de Python

Ahora respecto al código de Python, en este se emplean bibliotecas como sqlite3, pandas, matplotlib, y scipy.stats con el propósito  de calcular y hacer graficas de tipo estadísticos. Para esto se extraen los valores numéricos de proyecto.db, del cual se tiene dos variables `(variable_1 y variable_2)` de la tabla test_data. Además, en la conexión a la base de datos se utiliza SQL, ya que nos permite acceder a los datos para su posterior manipulación. Una vez conseguido los valores se almacenan en DataFrame y más adelante se observa parte del código:

```
Creación de H conn = sqlite3.connect('proyecto.db')
query = "SELECT variable_1, variable_2 FROM test_data"
data = pd.read_sql_query(query, conn)
conn.close()
```

Para la visualización de los datos se efectuaron histogramas para ambas variables y con base a eso, se le agregan líneas de mejor ajuste. Es importante mencionar que los histogramas son de bastante ayuda debido a que permiten observar de una mejor manera la forma de la distribución de los cifras, así se identifican las características como la simetría, la dispersión y posibles sesgos. El siguiente código ilustra cómo se generaron los histogramas:

```
plt.hist(data['variable_1'], bins=20, color='blue', alpha=0.7)
plt.hist(data['variable_2'], bins=20, color='green', alpha=0.7)
```

En el caso del ajuste de distribución, se aprecia como en la línea de código faculta evaluar si los datos siguen patrones específicos:

```
mu_1, std_1 = norm.fit(data['variable_1'])
loc_2, scale_2 = expon.fit(data['variable_2'], floc=0)
```

Luego, está el cálculo de los momentos estadísticos para las dos variables. Para alcanzar la media, varianza, desviación estándar, asimetría (skewness) y curtosis. Este contenido genera una descripción cuantitativa de las distribuciones y de igual forma se muestra cómo se realiza en el código:

```
mea_1 = data['variable_1'].mean()
var_1 = data['variable_1'].var()
skew_1 = skew(data['variable_1'])
kurt_1 = kurtosis(data['variable_1'])
``` 

