# Data Challenge Mayo 2020

## Descripción de proyecto y objetivo

-- Uri
1 o 2 parrafos

### Cararísticas técnicas

-- Julio

#### Lenguaje de programación

 - [Python 3.7](https://www.python.org/)

#### Librerías

- [Pandas 1.0.3](https://pandas.pydata.org/)

    Uso para lectura y manipulación de los datos como objetos **DataFrame**.

- [NumPy 1.18](https://numpy.org/)

    Uso para modificación, identificación y manejo de los datos.

- [SciPy 1.4](https://docs.scipy.org/)

    Uso para obtención de datos estadísticos *T-Test* y prubeas de normalidad *Shapiro-Wilk*.

- [SciKit-Learn 0.22](https://scikit-learn.org/stable/)

    Uso para selección de variables; procesamiento, imputación y trasformación de datos; entranmiento y selcción del modelo. (*Random Forest*, *Logistic Regression*)

- [MatPlotLib 3.2.1](https://matplotlib.org/)

    Uso para visualización de los datos, su comportamiento y su correlación.

- [Seaborn 0.10](https://seaborn.pydata.org/)

    Uso para visualización de los datos en un mapa de calor.


#### IDE de ejecución

- [Anaconda - Spyder 4.0](https://www.spyder-ide.org/)

## Descripción de datos

-- Enrique
``` 
/directorio diccionario/
```

base_1 contiene X, su granularidad es a nivel #SOL

¿Que contienen cada dataset? (de manera general, mencionar ubicación del diccionario)

## Carga y unión de datos

-- Enrique
¿Como se unen y con que granularidad?
Aquí estaría bien describimos las decisiones arbitrarias

## Generación de modelos

La generación de modelos se basó en el entrenamiento de **Random Forest** y **Logistic Regression** para cada mes definido en un rango de fecha mínima y fecha máxima de la variable **MES_COTIZACION**. La información de entrenamiento *train* se define como toda aquella que sea anterior a la fecha seleccionada. La información de prueba *test* se define como la misma que se esté seleccionando:
``` python
for month in dt_range:   
    train_temp = train[train.MES_COTIZACION <= month]
    test_temp  = test[test.MES_COTIZACION == month]
```
Teniendo el subset de datos de entrenamiento, se procesaran en una etapa llamada ***preprocess*** definida en un *Pipeline*, el cual se divide en 3 etapas:

- ***feature_select***: Desestimación de variables que fueron designadas por la selección de variables como "poco informativas" para el entramiento del modelo. Se usaron 3 métodos estadísticos:
    * Desestimación por porcentaje de nulos.
    * Desestimación por Coeficiones de Correlación de Person.
    * Desestimación por T-Test.

- ***numeric_transformer***: Adpatación y ajuste de datos numéricos en dos etapas:
    * Imputación de la mediana a registros nulos.
    * Estandariazción de datos removiendo la media y escalando a la variación unitaria.

- ***categorical_transformer***: Ajuste y codificación de datos categóricos en dos etapas:
    * Imputación de un valor constante *missing* a registros nulos.
    * Aplicar *OneHotEncoder* para cada categoría.

### Selección de variables

#### 1. Combinación variable numérica con variable numérica

-- Enrique

#### 2. Combinación variable numérica con variable categórica

-- Uri
``` python
import pandas as pd
```

#### 3. Combinación variable categórica con variable categórica

-- Julio

## Evaluación del modelo

### Crossvalidation y ParamGrid

-- Uri

accuracy
neg log loss

### Métricas alternas de evaluación

-- Enrique
tp fn tn fp
auc

## Conclusión

-- Uri


## Trabajo a futuro

-- Julio

Cosas por probar o que claramente son mejorables

## Contacto

Nombre y mail

Julio
Enrique
Uri
