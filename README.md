
# Clasificación Predictiva Ordinal de la Popularidad de Noticias Digitales mediante Redes Bayesianas

Este repositorio contiene el desarrollo completo del proyecto académico realizado para la evaluación práctica de la asignatura **Modelización de Datos Complejos**, dentro del **Grado de Estadística Aplicada**.

El objetivo académico fue la familiarización y explotación de las posibilidades de las Redes Bayesianas aplicadas a un problema real de interés propio, centralizando el trabajo en el estudio de diversas propuestas de modelos y la comparación y validación de los mismos para alcanzar el clasificador "óptimo" en algún sentido, bajo nuestros propios criterios adaptados al contexto de los datos.

Con este fin, se decidió emplear la oportunidad para predecir la popularidad de noticias digitales publicadas por **Mashable**, utilizando el dataset público **Online News Popularity**.

------------------------------------------------------------------------

## Resumen de las Decisiones Metodológicas Clave

### 1. Preprocesamiento, Depuración y Construcción de Variables

-   **Depuración de la Muestra**: Se identificaron y eliminaron **1.181 observaciones sin contenido textual** (`n_tokens_content = 0`), correspondientes a errores de recopilación o artículos exclusivamente multimedia. La muestra final de trabajo quedó en **38.463 observaciones válidas**.
-   **Variable Objetivo Ordinal (`popularidad`)**: Se adaptó a las necesidades del ejercicio discretizándola mediante un criterio que permitiera interpretar lógicamente sus clases, tratando también de corregir su fuerte asimetría:
    1.  Se eliminaron temporalmente los *outliers* mediante el método del **Rango Intercuartílico (RIC)**.
    2.  Se normalizó la variable dividiendo por el valor máximo no atípico, escalando los artículos estándar en un rango $[0, 100]$ y dejando los *outliers* con puntuaciones $> 100$.
    3.  Se crearon 4 niveles de popularidad ordinales: **`muy_baja`** [0, 20), **`baja`** [20, 50), **`media`** [50, 90) y **`alta`** $\ge 90$.
-   **Creación de Variable Síntesis (`channel`)**: Se condensaron 6 variables *dummy* independientes en una única variable categórica con 7 niveles (`business`, `entertainment`, `lifestyle`, `socmed`, `tech`, `world` y `other`) para simplificar la estructura bayesiana.
-   **Discretización por Cuantiles (Modificada)**: Las 12 predictoras continuas restantes se categorizaron en 3 niveles (`bajo`, `medio`, `alto`) fijando el rango intercuartílico $[Q_1 (25\%), Q_3 (75\%)]$ como la franja "normal/media" (50% de la distribución). Se aplicaron ajustes manuales específicos para variables con alta concentración de ceros (como `num_videos` o `is_weekend`).

### 2. Selección Automática de Variables (Markov Blanket + Bootstrap)

Para evitar incorporar ruido y garantizar la parsimonia del modelo, se aplicó un algoritmo **Markov Blanket** sobre DAGs aprendidos por *Hill-Climbing* ($20$ inicios aleatorios y $5$ perturbaciones): 

- Se ejecutó un **Bootstrap no paramétrico con 100 iteraciones**.

- Se evaluaron dos criterios de puntuación estructural: 
* **Criterio BIC**: Seleccionó una **base reducida de 5 predictoras** clave: `channel`, `is_weekend`, `kw_avg_avg_cat` (popularidad histórica de palabras clave), `num_hrefs_cat` y `n_tokens_content_cat`.
* **Criterio AIC**: Resultó ser más permisivo y conservó las **14 predictoras candidatas (base completa)**.

------------------------------------------------------------------------

## Arquitectura de Modelos y Evaluación

Se diseñó un experimento factorial comparando **12 combinaciones posibles**: 3 estructuras x 2 criterios de decisión x 2 conjuntos de datos (reducido vs. completo).

1.  **Estructuras probadas**:
    -   **Naive Bayes (NB)**: Asume independencia condicional entre predictoras dada la clase.
    -   **Tree-Augmented Naive Bayes (TAN)**: Relaja el supuesto de independencia permitiendo dependencias tipo árbol entre las predictoras.
    -   **Modelo tipo Prognosi**: Restringe los arcos para respetar la lógica temporal (las características del artículo son anteriores a la interacción de los usuarios).
2.  **Criterios de Clasificación**: Se propuso emplear ambos para determinar si el teóricamente adecuado realmente mejoraba la capacidad predictiva:
    -   **MAP** (*Maximum A Posteriori*): Regla estándar multi-clase.
    -   **Ord-MAP**: Adaptación específica que incorpora el orden natural de las categorías ordinales.
3.  **Mecanismo de Validación**: $k$-Fold Cross-Validation Estratificado ($k=10$), garantizando que los 10 *folds* mantuvieran exactamente la misma proporción de clases de popularidad para evitar sesgos por desbalance.

### Métricas Seleccionadas

Debido al desbalance de clases (31.5% `muy_baja`, 43.6% `baja`, 12.1% `media`, 12.9% `alta`) y a la naturaleza ordinal, la métrica principal de decisión fue el **AMAE** (*Adjusted Mean Absolute Error*), respaldada por *Balanced Accuracy*, *MAE* y *Accuracy*.

------------------------------------------------------------------------

## Selección del Modelo Final

Para validar qué modelo era óptimo en términos de AMAE, se aplicó una batería de tests de hipótesis sobre los vectores apareados de AMAE ($k=10$ *folds*).

Debido al gran volumen de comparaciones a realizar, se decidió establecer un orden lógico y argumentado que nos permitiera proceder más detenidamente: como las 12 propuestas provienen de 2 conjuntos de predictoras diferentes, primero se determinó la preferida para cada conjunto y posteriormente se enfrentaron estas dos para determinar la ganadora.

1.  **Test Global de Friedman**: Confirmó la existencia de diferencias globales significativas entre los modelos ($p < 0.0001$ tanto en la base reducida como en la completa).

2.  **Test de Normalidad de Shapiro-Wilk**: Evaluó la distribución de las diferencias apareadas dos a dos para determinar la elección de tests paramétricos o no paramétricos.

3.  **Comparaciones Apareadas Unilaterales (**$t$-test / Wilcoxon) con **Corrección de Holm para Multiplicidad**:
    -   **En la base reducida**: **Naive Bayes con Ord-MAP** obtuvo un AMAE medio de $0.9445$, siendo **significativamente inferior** ($p < 0.001$) a todos los demás clasificadores.
    -   **En la base completa**: El modelo Naive Bayes Ord-MAP ($AMAE = 0.9474$) y el modelo TAN Ord-MAP ($AMAE = 0.9493$) no mostraron diferencias estadísticamente significativas entre sí ($p = 0.235$). Se seleccionó el modelo Naive Bayes Ord-MAP por parsimonia, pero se mantuvo el Tree-Augmented Naive Bayes como modelo secundario.
    -   **Comparación Final (Reducida vs. Completa)**: Un test de Wilcoxon apareado entre los dos modelos ganadores descartó diferencias significativas ($p = 0.2158$).

**Modelo Principal Seleccionado**: **Naive Bayes con regla Ord-MAP sobre la base reducida (BIC)**, por el principio de **parsimonia**.

------------------------------------------------------------------------

## Inferencia, Hallazgos y Análisis de Resultados

Una vez seleccionado el modelo principal (**Naive Bayes con Ord-MAP** sobre la base reducida), se realizó un análisis exhaustivo de inferencia y predicción para comprender el comportamiento del clasificador y extraer conclusiones sobre los datos.

### 1. Capacidad Discriminativa de las Variables

Para evaluar qué variables aportaban mayor información al modelo, se cuantificó la diferencia máxima y mínima entre las probabilidades condicionadas de cada categoría frente a los niveles de popularidad:

* **`kw_avg_avg_cat` (Popularidad histórica de palabras clave)**: Fue la variable con **mayor capacidad discriminativa** ($0.204$). Los artículos con palabras clave de impacto alto multiplican notablemente su probabilidad de alcanzar categorías superiores.

* **`is_weekend` (Publicación en fin de semana)**: Mostró un fuerte poder discriminativo ($0.145$). Publicar en fin de semana reduce la probabilidad de popularidad `muy_baja` y desplaza la distribución hacia los niveles `media` y `alta`.

* **`num_hrefs_cat` y `channel`**: Presentaron capacidades intermedias ($\approx 0.106$). Destaca que canales como **`other`**, **`socmed`** y **`lifestyle`** concentran una mayor proporción de artículos altamente populares frente a temáticas como `world` o `entertainment`.

* **`n_tokens_content_cat` (Longitud)**: Fue la variable con **menor impacto** ($0.059$), mostrando una distribución muy homogénea (alrededor del 50% en longitud media) de forma transversal en todas las clases de popularidad.

### 2. Estudio de Probabilidades Posteriores y Evaluación de Perfiles

Mediante el uso de consultas de probabilidad condicionada $P(\text{popularidad} = y \mid X_j = x)$, se analizó cómo cambiaban las expectativas de difusión ante perfiles concretos: 

* **Perfiles Favorables**: Combinar canales de alta difusión (`socmed`), publicación en fin de semana, muchos enlaces y palabras clave de alta popularidad desplaza la probabilidad posterior hacia las clases `media` y `alta`.

* **Perfiles Desfavorables**: Artículos en canales como `world`, publicados entre semana y con bajo historial de palabras clave acumulan una probabilidad dominante en la categoría `muy_baja` ($\approx 58\%$).

* **Análisis de Casos Nuevos (Combinaciones No Vistas)**: Se probaron 15 combinaciones teóricas que no aparecían en el entrenamiento (3.97% del total posible). Aunque el modelo tiende a predecir mayoritariamente la clase `baja` en estos escenarios inéditos, esto no es un fallo, sino una consecuencia lógica: debido al fuerte desbalance del dataset original y al suavizado aplicado, ante una combinación que no conoce, el modelo adopta una postura conservadora. Aun así, ofrece respuestas estables y coherentes.

### 3. Contextualización del Rendimiento Predictivo

-   **Métrica AMAE (**$0.9445$): Al tratarse de un problema con 4 clases ordinales, un AMAE inferior a $1.0$ indica que, en promedio, **el error del modelo es menor a una categoría de distancia**.

-   **Sesgo hacia la Clase Mayoritaria**: La matriz de confusión reveló una clara tendencia a sobreestimar la clase `baja` (la más frecuente), un comportamiento esperable dado el desbalance natural del dataset y la complejidad inherente a la predicción de comportamientos e interacciones sociales (fenómenos sociológicos donde la variabilidad es extrema y difícil de capturar en su totalidad mediante variables puramente editoriales).

------------------------------------------------------------------------

## Análisis Secundarios

-   **Estructura TAN Secundaria**: Reveló dependencias clave entre predictoras: `channel` actúa como nodo padre directo de `is_weekend` y `kw_avg_avg_cat`; mientras que las variables de **subjetividad y polaridad** forman un clúster interdependiente aislado del resto.

-   **Análisis Bivariante Complementario**: Contrastes de Chi-cuadrado y $V$ de Cramer ($V \approx 0.16$ para palabras clave y canal) confirmaron la dependencia estadística significativa de todas las predictoras seleccionadas con la respuesta ($p < 0.0001$).

------------------------------------------------------------------------

## Limitaciones y Propuestas de Mejora

Como cierre del proyecto, se identificaron algunas áreas de mejora técnica y metodológica de cara a futuros trabajos: 

- **Optimización del Código**: El flujo de trabajo presentó duplicación de código entre modelos y sobreescritura de objetos intermedios para simplificar la ejecución. En el futuro, se vectorizaría y estructuraría mediante funciones auxiliares y reutilizables.

- **Criterio de Selección Markov Blanket**: El uso del criterio AIC resultó muy permisivo (conservando las 14 variables). Como mejora, se propone establecer un umbral basado en un porcentaje estricto de apariciones en el *Bootstrap* en lugar de una única aparición.

- **Alternativas No Paramétricas o Lineales**: Dado que la base de datos original era mayoritariamente numérica, el proceso de discretización supuso una pérdida de información. Sería interesante contrastar estos resultados con modelos lineales generalizados u otras técnicas de datos continuos.
