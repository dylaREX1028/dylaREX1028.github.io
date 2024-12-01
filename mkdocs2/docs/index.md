# Introducción

En el presente proyecto tiene la finalidad de analizar un conjunto de datos utilizando el lenguaje de programación Python. Los datos se recopilan de una base de datos SQLite, se procesan y se utilizan técnicas estadísticas para ajustarlas a los modelos según lo estudiado en clase. Este análisis se hace de dos bases de datos, el primer recopilado es por 12h `(proyecto.db)` y el segundo por 24h `(final.db)`.

Respecto al código de Python, en este se emplean bibliotecas como sqlite3, pandas, matplotlib, scipy.stats, numpy.polynomial.polynomial, numpy y fitter, con el propósito  de calcular y hacer graficas de tipo estadísticos. Para esto se extraen los valores numéricos de la base de datos. En caso del primero se realiza con `proyecto.db`, del cual se tiene dos variables `(variable_1 y variable_2)`, y en el segundo mediante `final.db` , en este se posee `(variable_1, timestamp y sunligth)` de la tabla test_data. Además, en la conexión a la base de datos se utiliza SQL, ya que nos permite acceder a los datos para su posterior manipulación. 




