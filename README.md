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

Para determinar si el target (variable categórica) puede dividir los valores de variables numéricas en dos grupos con medias que diferentes de manera estadísticamente significante, empleamos la prueba t de Student como se muestra a continuación.

``` python
droped_ttest_cols = []         
# * Evaluar normalidad "skewness"
target = train_temp[label]
t_sel = [0] * len(train_num.columns) # señala qué variables pueden ayudar a predecir target
t_ctr = 0 # contador
for col in train_num.columns:
    # Shapiro-Wilk test
    stat, p = shapiro(train_num[col])
    if p > 0.05: # no se rechaza la H0 según la cual la distribución de estos datos es similar a la gaussiana
        # t-test
        # separación de datos según la aceptación del crédito
        t0 = train_num[col][target == 0]
        t1 = train_num[col][target == 1]
        stat, p = ttest_ind(t0, t1, nan_policy = "omit", equal_var = False)
        # print('T-statistic={:.3f}, p={:.3f}'.format(stat, p))
            
        if p < 0.05: # se rechaza la H0 según la cual las medias de t0 y t1 no difieren significativamente
            t_sel[t_ctr] = 1
        else:
            droped_ttest_cols.append(col)
            pass
    t_ctr += 1
        
t_selec = pd.DataFrame(t_sel, index = train_num.columns)
```
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
