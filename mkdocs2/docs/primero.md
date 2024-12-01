# Líneas de código importantes para proyecto.db


El primer código `tasks.py` posee una herramienta llamada Celery, la cual se encarga de ejecutar las tareas en segundo plano. Para esta ocación, se llega a descargar los datos de una API, se prosesa y se guardan en una tabla de base de datos llamada `TestData`.

```python

from celery import Celery
from celery.schedules import timedelta
from datetime import datetime
import requests
import json
import configparser
import logging

from models import session, TestData


# Crear "app" de Celery
app = Celery("tasks", broker="redis://localhost")


# Configurar las tareas de Celery
@app.task
def test_task(url, group):
    """Descarga datos de una API y los almacena
    en la tabla de ejemplo de una base de datos.

    Parameters
    ----------
    url : str
        URL de la API.
    group : str
        Número de grupo del proyecto.

    Returns
    -------
    str
        Mensaje de éxito.
    """
    params = {"grupo": int(group)}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = json.loads(response.text)

        timestamp = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
        sample_size = data["sample_size"]
        for sample in range(sample_size):
            record = TestData(
                group=group,
                timestamp=timestamp,
                variable_1=data["variable_1"][sample],
                variable_2=data["variable_2"][sample],

            )
            session.add(record)
            session.commit()
        return "¡Hola mundo!"
    else:
        logging.error(f"Error {response.status_code}: {response.text}")
        return "Algo falló en la solicitud de datos."


@app.task
def schedule_task():
    return "¡Hola mundo cada 60 minutos!"


# ----------
# Configurar aquí las tareas de Celery para el procesamiento de los datos
# ----------

# Datos de configuración
config = configparser.ConfigParser()
config.read("proyecto.cfg")
url = config["api"]["url"]
group = config["api"]["group"]
period = int(config["scheduler"]["period"])

# Configurar el planificador de tareas de Celery
app.conf.beat_schedule = {
    "test-schedule": {
        "task": "tasks.test_task",
        "args": (url, group),
        "schedule": timedelta(seconds=period),
    },
    "test-schedule-task": {
        "task": "tasks.schedule_task",
        "schedule": timedelta(minutes=60),
    },
}
```



Ahora con el segundo código, el cual se ocupa se manejar la base de datos y puntualiza la forma de alamacenimiento de los datos en la tabla `TestData`. Además, de conectar la base de datos, esta crea una tabla si fuera el caso que no existiera. Por lo tanto, prepara el entorno para guardar información.


```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import configparser


# Crear la clase base de la tabla
class Base(DeclarativeBase):
    pass


# Datos de configuración
config = configparser.ConfigParser()
config.read("proyecto.cfg")
db = config["db"]["db"]
if db == "sqlite":
    system = config["db"]["sqlite"]
elif db == "postgresql":
    system = config["db"]["postgresql"]


# Definir los modelos
class TestData(Base):
    __tablename__ = "test_data"

    id = Column(Integer, primary_key=True)
    group = Column(String)
    timestamp = Column(DateTime)
    variable_1 = Column(Integer)
    variable_2 = Column(Float)


# Crear la conexión a la base de datos SQLite3 o PostgreSQL
engine = create_engine(system)
Session = sessionmaker(bind=engine)
session = Session()

# Crear la(s) tabla(s) en la base de datos
Base.metadata.create_all(engine)
```


En el caso de la base de datos, se pueden descaragar en el siguiente link:
- [Descargar base de datos](archivos/proyecto.db)


Una vez conseguido los valores se almacenan en DataFrame y más adelante se observa parte del código:

```python
Creación de H conn = sqlite3.connect('proyecto.db')
query = "SELECT variable_1, variable_2 FROM test_data"
data = pd.read_sql_query(query, conn)
conn.close()
```

Para la visualización de los datos se efectuaron histogramas para ambas variables y con base a eso, se le agregan líneas de mejor ajuste. Es importante mencionar que los histogramas son de bastante ayuda debido a que permiten observar de una mejor manera la forma de la distribución de los cifras, así se identifican las características como la simetría, la dispersión y posibles sesgos. El siguiente código ilustra cómo se generaron los histogramas:

```python
plt.hist(data['variable_1'], bins=20, color='blue', alpha=0.7)
plt.hist(data['variable_2'], bins=20, color='green', alpha=0.7)
```

En el caso del ajuste de distribución, se aprecia como en la línea de código faculta evaluar si los datos siguen patrones específicos:

```
mu_1, std_1 = norm.fit(data['variable_1'])
loc_2, scale_2 = expon.fit(data['variable_2'], floc=0)
```

Luego, está el cálculo de los momentos estadísticos para las dos variables. Para alcanzar la media, varianza, desviación estándar, asimetría (skewness) y curtosis. Este contenido genera una descripción cuantitativa de las distribuciones y de igual forma se muestra cómo se realiza en el código:

```python
mea_1 = data['variable_1'].mean()
var_1 = data['variable_1'].var()
skew_1 = skew(data['variable_1'])
kurt_1 = kurtosis(data['variable_1'])
```






















