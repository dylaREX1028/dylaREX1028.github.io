# Líneas de código importantes


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

 De igual manera lo puede descargar mediante este link:
- [Descargar código Python](archivos/tasks.py)


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

 De igual manera lo puede descargar mediante este link:
- [Descargar código Python](archivos/models.py)



En el caso de la base de datos, se pueden descaragar en el siguiente link:
- [Descargar base de datos](archivos/proyecto.db)

