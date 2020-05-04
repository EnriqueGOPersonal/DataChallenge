# Data Challenge Mayo 2020

## Descripción de proyecto y objetivo

-- Uri
1 o 2 parrafos

### Cararísticas técnicas

-- Julio
python v
librerias
software

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

-- Julio
Se genera uno por mes con el fin de... tomando los datos hasta...

### Selección de variables

#### 1. Combinación variable numérica con variable numérica

Coeficiente de correlación

Es una medida de dependencia lineal entre dos variables aleatorias cuantitativas. El valor del índice de correlación varía en el intervalo [-1,1], indicando el signo el sentido de la relación, y donde mayor sea el valor absoluto del índice de correlacion, existe más dependencia lineal.

En la selección de variables se emplea este índice eliminar una de cada par de variables que dependan en alto grado, pues una de las dos no aportará información nueva al modelo.

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
Luis Enrique García Orozco [luisenrique.garcia.orozco@bbva.com]
Uri
