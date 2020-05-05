# Data Challenge Mayo 2020

## Descripción de proyecto y objetivo

Los créditos que ofrecen los bancos varían ampliamente por tasa, importe y plazo. Para incrementar el rendemiento de préstamos crediticios, los bancos deben identificar qué condiciones bancarias y demográficas incrementan la probabilidad de que el cliente acepte una oferta de crédito. 

En este proyecto utilizamos la riqueza informacional que BBVA Perú proporciona sobre clientes y sus transacciones para explotarla a través de modelos de clasificación de Machine Learning que ayuden a identificar las condiciones de préstamo bajo las que un cliente aceptará una oferta crediticia. 

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

Los datos empleados se encuentran descritos a profundidad en el archivo situado en la ruta:

``` 
\DataChallenge\data\"Diccionario Datos Challenge.xlsx"
```

Los datos empleados se encuentran en la ruta:

``` 
\DataChallenge\data\"
```

Conteniendo cada uno:

**Fichero 1** es la Base de cotizaciones, tiene una granularidad a nivel -mes de cotización / código de solicitud / código de cliente-.

**Fichero 2** es la Base sociodemográfica + Digital, tiene una granularidad a nivel -mes de cotización / código de cliente-. 

**Fichero 3** es la Base de productos BBVA, tiene una granularidad a nivel -mes de cotización / mes de registro de datos / código de cliente-.

**Fichero 4** es la Base de saldos en el Sistema Financiero, tiene una granularidad a nivel -mes de cotización / mes de registro de datos / código de cliente / código de entidad bancaria-.

**Fichero 5** es la Base de consumos con tarjeta, tiene una granularidad a nivel -mes de cotización / mes de registro de datos / código de cliente-.

## Carga y unión de datos

Los datos de las tablas descritas en la sección Descripción de datos fueron unidos en una tabla 
de granularidad - mes de cotización / código de cliente-.

En caso de que existieran registros repetidos para la combinación -mes de cotización /código de cliente- se generaron distintos tipos de agregaciones de acuerdo a el tipo de variable.

Variables Categóricas

Conteo de cantidad de registros de cada valor de la variable. Por ejemplo, si en un mismo mes de cotización para un mismo cliente se tienen dos solicitudes con garantía "A" y una solicitud con garantía "B", se genera una columna llamada "garantia_A_cnt" con valor de 2 y una columna llamada "garantia_B_cnt" con valor de 1. El resto de columnas generadas para el resto de valores observados se mostrarán como 0.

Variables Numéricas

Promedio, máximo y mínimo de los valores observados para dicha la variable. Por ejemplo, si en un mismo mes de cotización para un mismo cliente se tienen dos importes uno con valor de 200 y otro con valor de 300, se generarán las columnas "importe_avg" con valor de 250, "importe_max" con valor de 300 e "importe_min" con valor de 200. Cuando no existen observaciones se mostrará 0 en todas las columnas.

Variable Dependiente (Categórica):

Como se desea conservar la variable con valores de 1 y 0 únicamente, cuando aparecen dos solicitudes indistinguibles el mismo mes, se combinan en una sola solicitud que toma el valor de 1 si alguna de las dos fue aceptada y de 0 si ninguna lo fue. Adicionalmente se crea una columna con la cantidad de solicitudes encontradas para la el mismo mes de cotización para un mismo cliente.

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

Coeficiente de correlación

Es una medida de dependencia lineal entre dos variables aleatorias cuantitativas. El valor del índice de correlación varía en el intervalo [-1,1], indicando el signo el sentido de la relación, y donde mayor sea el valor absoluto del índice de correlacion, existe más dependencia lineal.

En la selección de variables se emplea este índice eliminar una de cada par de variables que dependan en alto grado (valor absoluto del indice > 0.9), pues una de las dos no aportará información nueva al modelo.

#### 2. Combinación variable numérica con variable categórica

Para determinar si el target (variable categórica) puede dividir los valores de variables numéricas en dos grupos con medias que diferentes de manera estadísticamente significante, empleamos la prueba t de Student como se muestra a continuación.

En este proyecto fijamos el umbral de significancia estadística en valores p < 0.05. Sólo las variables con valores distribuidos de manera gaussiana fueron sometidos a la prueba t de Student. La normalidad fue evaluada por medio de la prueba de Shapiro-Wilk.

La implicación de un valor p < .05 es que la variable numérica puede dividirse en dos grupos que difieren en su media y que están vinculados con uno de los dos valores del target. Esto sugiere que tal variable es útil para la predicción del target.

#### 3. Combinación variable categórica con variable categórica

-- Julio

## Evaluación del modelo

### Cross-validation y ParamGrid



```python
param_grid_lr = {
    'classifier__C': [0.01, 0.001, 0.1, 1.0],
}

param_grid_rf = {
    'classifier__n_estimators': [150, 200]
}

lr_clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter = 500))])    
    

rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_jobs = -1))])    
    
grid_search_lr = GridSearchCV(lr_clf, param_grid_lr, cv = 5, n_jobs = -1, scoring = "accuracy")
grid_search_rf = GridSearchCV(rf_clf, param_grid_rf, cv = 5, n_jobs = -1, scoring = "accuracy")
    
features = [col for col in train_temp.columns if col != label]
x_train = train_temp[features]
y_train = train_temp[label]

grid_search_rf.fit(x_train, y_train)
grid_search_lr.fit(x_train, y_train)
```
accuracy con probabilidad de 50% de threshold: 0.63692
log loss: 0.6420

### Métricas alternativas de evaluación

El desempeño del modelo seleccionado se puede medir de manera alternativa a través de medidas como el area bajo la curva ROC (conocida común mente como AUC ROC) o el índice de Gini, que es equivalente a ```(2 * AUC_ROC) - 1```.

El mejor modelo reportó una AUC ROC de 0.6858, y por lo tanto un índice de Gini de 0.3716

## Conclusión

-- Uri


## Trabajo a futuro

-- Julio

Cosas por probar o que claramente son mejorables

## Contacto

Julio

Luis Enrique García Orozco (<luisenrique.garcia.orozco@bbva.com>)

Uri
