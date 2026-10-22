
####################################
# Manuela López Cambrón (1673688)
# Abril P. Lozano Díaz (1670769)
# Àlex Hinojal Rodríguez (1670853)
####################################

## -------------------------------------------------------------------------------------------------------------
# LIBRERÍAS
# bnlearn para aprender, ajustar y representar redes bayesianas
# Se utilizan las funciones hc(), mb(), naive.bayes(), tree.bayes(), bn.fit(),
# as.grain() y graphviz.plot()
library(bnlearn)

# gRain para realizar inferencia sobre las redes bayesianas
# Se utilizan la funciones setEvidence(), querygrain() y predict() 
# sobre objetos gRain
library(gRain)

# Rgraphviz como soporte gráfico para graphviz.plot() de bnlearn
library(Rgraphviz)

# ggplot2 para gráficos de barras
# Se utilizan las funciones ggplot(), aes(), ggsave(), theme_minimal(), entre
# otras típicas de ggplot2
library(ggplot2)

# patchwork para juntar gráficos en uno solo
# Se utilizan las funciones wrap_plots() y plot_annotation()
library(patchwork)

# bnlearn para aprender y representar la red TAN AIC
# Se utilizan las funciones tree.bayes(), arcs() y graphviz.plot()
library(bnlearn)

set.seed(123)

## -------------------------------------------------------------------------------------------------------------
# CARGAR BASE DE DATOS
data <- read.csv("OnlineNewsPopularity.csv")


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# PREPARACIÓN DE LOS DATOS
############################################################################## 

## Codificación variable target
shares <- data$shares

Q1 <- quantile(shares, 0.25)
Q3 <- quantile(shares, 0.75)

RIC <- Q3 - Q1

shares2 <- shares[shares >= (Q1 - 1.5 * RIC) & shares <= (Q3 + 1.5 * RIC)]

shares2 <- shares/max(shares2)*100

data$popularidad <- cut(
  shares2,
  breaks = c(-Inf, 20, 50, 90, Inf),
  labels = c("muy_baja", "baja", "media", "alta"),
  right = FALSE
)

tabla_popularidad <- table(data$popularidad)

cat("\nVariable: popularidad")

tabla_popularidad

## Tranformación variables predictivas
### Variable "channel"
data$channel <- "other"

data$channel[data$data_channel_is_lifestyle == 1] <- "lifestyle"
data$channel[data$data_channel_is_entertainment == 1] <- "entertainment"
data$channel[data$data_channel_is_bus == 1] <- "business"
data$channel[data$data_channel_is_socmed == 1] <- "socmed"
data$channel[data$data_channel_is_tech == 1] <- "tech"
data$channel[data$data_channel_is_world == 1] <- "world"

data$channel <- factor(data$channel)

tabla_channel <- table(data$channel)

cat("\nVariable: channel")

tabla_channel

## Depuración
num_0tokens <- sum(data$n_tokens_content == 0)

cat("\nNúmero de casos con 0 tokens:", num_0tokens)

data <- data[data$n_tokens_content > 0, ] # eliminamos casos sin valor informativo (o possible distorsión de otras variables)

dim_depuracion <- dim(data)

cat("\nCasos finales: ", dim_depuracion[1])


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# SELECCIÓN DE VARIABLES 1 - criterio lógico
############################################################################## 

vars_finales <- c(
  "popularidad",
  
  "channel",
  "is_weekend",
  
  "n_tokens_title",
  "n_tokens_content", # he quitado n_nonstop_words pq no funciona bien, es rara
  "num_hrefs",
  "num_imgs",
  "num_videos",

  "kw_avg_avg",
  
  "global_subjectivity",
  "avg_positive_polarity", 
  "avg_negative_polarity", 
  
  "title_subjectivity",
  "title_sentiment_polarity",
  "abs_title_sentiment_polarity" # pq mide tambine la intesidad de la emocion
)

data <- data[vars_finales]


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# CATEGORIZACIÓN DE VARIABLES
############################################################################## 

## ANÁLISIS DESCRIPTIVO (preliminiar, para categorizar)

resumen_variable <- function(x) { # específico para ver más detalles de distribucion
  c(
    n = length(x),
    min = min(x, na.rm = TRUE),
    p01 = quantile(x, 0.01, na.rm = TRUE),
    p05 = quantile(x, 0.05, na.rm = TRUE),
    p10 = quantile(x, 0.10, na.rm = TRUE),
    Q1 = quantile(x, 0.25, na.rm = TRUE),
    mediana = median(x, na.rm = TRUE),
    media = mean(x, na.rm = TRUE),
    Q3 = quantile(x, 0.75, na.rm = TRUE),
    p90 = quantile(x, 0.90, na.rm = TRUE),
    p95 = quantile(x, 0.95, na.rm = TRUE),
    p99 = quantile(x, 0.99, na.rm = TRUE),
    max = max(x, na.rm = TRUE),
    ceros = sum(x == 0, na.rm = TRUE),
    pct_ceros = mean(x == 0, na.rm = TRUE) * 100
  )
}

variables_continuas <- c(
  "n_tokens_title",
  "n_tokens_content",
  "num_hrefs",
  "num_imgs",
  "num_videos",
  "kw_avg_avg",
  "global_subjectivity",
  "avg_positive_polarity",
  "avg_negative_polarity",
  "title_subjectivity",
  "title_sentiment_polarity",
  "abs_title_sentiment_polarity"
)

resumenes <- lapply(data[variables_continuas], resumen_variable)

for (v in variables_continuas) {
  cat("\n====================================\n")
  cat("Variable:", v, "\n")
  cat("====================================\n")
  print(round(resumenes[[v]], 3))

  par(mfrow = c(1, 2))
  
  hist(
    data[[v]],
    main = paste("Histograma de", v),
    xlab = v,
    col = "lightgray",
    border = "white"
  )
  
  boxplot(
    data[[v]],
    main = paste("Boxplot de", v),
    ylab = v,
    col = "lightgray"
  )
  
  par(mfrow = c(1, 1))
}

# ------------------------------------------------------------
# N_TOKENS_TITLE
# ------------------------------------------------------------

q_n_tokens_title <- quantile(
  data$n_tokens_title,
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$n_tokens_title_cat <- cut(
  data$n_tokens_title,
  breaks = c(-Inf, q_n_tokens_title[1], q_n_tokens_title[2], Inf),
  labels = c("corto", "medio", "largo"),
  right = TRUE
)

data$n_tokens_title_cat <- factor(
  data$n_tokens_title_cat,
  levels = c("corto", "medio", "largo")
)

# ------------------------------------------------------------
# N_TOKENS_CONTENT
# ------------------------------------------------------------

q_n_tokens_content <- quantile(
  data$n_tokens_content,
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$n_tokens_content_cat <- cut(
  data$n_tokens_content,
  breaks = c(0, q_n_tokens_content[1], q_n_tokens_content[2], Inf),
  labels = c("corto", "medio", "largo"),
  right = TRUE
)

data$n_tokens_content_cat <- factor(
  data$n_tokens_content_cat,
  levels = c("corto", "medio", "largo")
)

# ------------------------------------------------------------
# NUM_HREFS
# ------------------------------------------------------------

q_num_hrefs <- quantile(
  data$num_hrefs,
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$num_hrefs_cat <- cut(
  data$num_hrefs,
  breaks = c(-Inf, q_num_hrefs[1], q_num_hrefs[2], Inf),
  labels = c("pocos_ninguno", "moderados", "muchos"),
  right = TRUE
)

data$num_hrefs_cat <- factor(
  data$num_hrefs_cat,
  levels = c("pocos_ninguno", "moderados", "muchos")
)

# ------------------------------------------------------------
# NUM_IMGS
# ------------------------------------------------------------

data$num_imgs_cat <- cut(
  data$num_imgs,
  breaks = c(-Inf, 0, 1, 4, Inf),
  labels = c("sin_imagenes", "una", "varias", "muchas"),
  right = TRUE
)

data$num_imgs_cat <- factor(
  data$num_imgs_cat,
  levels = c("sin_imagenes", "una", "varias", "muchas")
)

# ------------------------------------------------------------
# NUM_VIDEOS
# ------------------------------------------------------------

data$num_videos_cat <- cut(
  data$num_videos,
  breaks = c(-Inf, 0, 1, Inf),
  labels = c("sin_videos", "uno", "varios"),
  right = TRUE
)

data$num_videos_cat <- factor(
  data$num_videos_cat,
  levels = c("sin_videos", "uno", "varios")
)

# ------------------------------------------------------------
# KW_AVG_AVG
# ------------------------------------------------------------

q_kw_avg_avg <- quantile(
  data$kw_avg_avg,
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$kw_avg_avg_cat <- cut(
  data$kw_avg_avg,
  breaks = c(-Inf, q_kw_avg_avg[1], q_kw_avg_avg[2], Inf),
  labels = c("baja", "media", "alta"),
  right = FALSE
)

data$kw_avg_avg_cat <- factor(
  data$kw_avg_avg_cat,
  levels = c("baja", "media", "alta")
)

# ------------------------------------------------------------
# GLOBAL_SUBJECTIVITY
# ------------------------------------------------------------

q_global_subjectivity <- quantile(
  data$global_subjectivity,
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$global_subjectivity_cat <- cut(
  data$global_subjectivity,
  breaks = c(-Inf, q_global_subjectivity[1], q_global_subjectivity[2], Inf),
  labels = c("objetiva", "intermedia", "subjetiva"),
  right = FALSE
)

data$global_subjectivity_cat <- factor(
  data$global_subjectivity_cat,
  levels = c("objetiva", "intermedia", "subjetiva")
)

# ------------------------------------------------------------
# AVG_POSITIVE_POLARITY
# ------------------------------------------------------------

q_avg_positive_polarity <- quantile(
  data$avg_positive_polarity,
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$avg_positive_polarity_cat <- cut(
  data$avg_positive_polarity,
  breaks = c(-Inf, q_avg_positive_polarity[1], q_avg_positive_polarity[2], Inf),
  labels = c("baja_nula", "media", "alta"),
  right = FALSE
)

data$avg_positive_polarity_cat <- factor(
  data$avg_positive_polarity_cat,
  levels = c("baja_nula", "media", "alta")
)

# ------------------------------------------------------------
# AVG_NEGATIVE_POLARITY
# ------------------------------------------------------------

q_avg_negative_polarity <- quantile(
  data$avg_negative_polarity,
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$avg_negative_polarity_cat <- cut(
  data$avg_negative_polarity,
  breaks = c(-Inf, q_avg_negative_polarity[1], q_avg_negative_polarity[2], Inf),
  labels = c("alta", "media", "leve_nula"),
  right = FALSE
)

data$avg_negative_polarity_cat <- factor(
  data$avg_negative_polarity_cat,
  levels = c("leve_nula", "media", "alta")
)

# ------------------------------------------------------------
# TITLE_SUBJECTIVITY
# ------------------------------------------------------------

q_title_subjectivity <- quantile(
  data$title_subjectivity[data$title_subjectivity > 0],
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$title_subjectivity_cat <- ifelse(
  data$title_subjectivity == 0,
  "objetivo",
  as.character(cut(
    data$title_subjectivity,
    breaks = c(0, q_title_subjectivity[1], q_title_subjectivity[2], Inf),
    labels = c("algo_subjetivo", "subjetividad_media", "subjetivo"),
    right = FALSE
  ))
)

data$title_subjectivity_cat <- factor(
  data$title_subjectivity_cat,
  levels = c("objetivo", "algo_subjetivo", "subjetividad_media", "subjetivo")
)

# ------------------------------------------------------------
# TITLE_SENTIMENT_POLARITY
# ------------------------------------------------------------

data$title_sentiment_polarity_cat <- ifelse(
  data$title_sentiment_polarity < 0,
  "negativo",
  ifelse(
    data$title_sentiment_polarity == 0,
    "neutro",
    "positivo"
  )
)

data$title_sentiment_polarity_cat <- factor(
  data$title_sentiment_polarity_cat,
  levels = c("negativo", "neutro", "positivo")
)

# ------------------------------------------------------------
# ABS_TITLE_SENTIMENT_POLARITY
# ------------------------------------------------------------

q_abs_title_sentiment_polarity <- quantile(
  data$abs_title_sentiment_polarity[data$abs_title_sentiment_polarity > 0],
  probs = c(0.25, 0.75),
  na.rm = TRUE
)

data$abs_title_sentiment_polarity_cat <- ifelse(
  data$abs_title_sentiment_polarity == 0,
  "sin_intensidad",
  as.character(cut(
    data$abs_title_sentiment_polarity,
    breaks = c(0, q_abs_title_sentiment_polarity[1], q_abs_title_sentiment_polarity[2], Inf),
    labels = c("intensidad_baja", "intensidad_media", "intensidad_alta"),
    right = FALSE
  ))
)

data$abs_title_sentiment_polarity_cat <- factor(
  data$abs_title_sentiment_polarity_cat,
  levels = c("sin_intensidad", "intensidad_baja", "intensidad_media", "intensidad_alta")
)


## -------------------------------------------------------------------------------------------------------------
### Base de datos después de la SELECCIÓN DE VARIABLES 1 (variables categorizadas)

vars_cat <- c(
  "popularidad",
  "channel",
  "is_weekend",
  
  "n_tokens_title_cat",
  "n_tokens_content_cat",

  "num_hrefs_cat",
  "num_imgs_cat",
  "num_videos_cat",
  "kw_avg_avg_cat",
  "global_subjectivity_cat",
  "avg_positive_polarity_cat",
  "avg_negative_polarity_cat",
  "title_subjectivity_cat",
  "title_sentiment_polarity_cat",
  "abs_title_sentiment_polarity_cat"
)


data_cat <- data[vars_cat]

data_cat[] <- lapply(data_cat, factor) # convertir todo a factor
cat("\n========================================\n")
cat("BASE DE DATOS - SELECCIÓN DE VARIABLES 1\n")
cat("========================================\n")
str(data_cat)

### Comprobación de frecuencias (absolutas y cruzadas con la respuesta)
cat("\n========================================\n")
cat("FRECUENCIAS DE NIVELES\n")
cat("========================================\n")

for (v in names(data_cat)) {
  
  if (v != "popularidad") {
    
    cat("\n========================================\n")
    cat("Variable:", v, "\n")
    cat("========================================\n")
    
    print(table(data_cat[[v]]))
    
    tab <- table(data_cat[[v]], data_cat$popularidad)
    print(tab)
    
    cat("\nPorcentajes por fila:\n")
    print(round(100 * prop.table(tab, margin = 1), 2))
  }
}


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# SELECCIÓN DE VARIABLES 2 - Markov Blanket con Bootstrap (No paramétric)
##############################################################################

# Haremos dos procesos uno utilizando criterio bic y otro aic

############################################################################## 
# MARKOV BLANKET - BIC
##############################################################################

# Variable objetivo
variable_objetivo <- "popularidad"

# Variables candidatas, excluyendo la variable objetivo
variables_predictoras <- setdiff(
  names(data_cat),
  variable_objetivo
)

# Número de muestras bootstrap
R_bootstrap <- 100

# Contador: suma 1 cada vez que una variable aparece en el MB de popularidad
contador_mb <- rep(
  0,
  length(variables_predictoras)
)

names(contador_mb) <- variables_predictoras

#### BUCLE BOOTSTRAP

for (b in 1:R_bootstrap) {
  
  cat("\nRepetición bootstrap:", b, "de", R_bootstrap, "\n") # esto lo quitare pero de momento va bine para ver el proceso
  
  # 1. Generar muestra bootstrap de data_cat
  indices_bootstrap <- sample(
    1:nrow(data_cat),
    size = nrow(data_cat),
    replace = TRUE
  )
  
  data_bootstrap <- data_cat[indices_bootstrap,]
  
  # 2. Mantener los niveles originales de los factores (por si en alguna muetsra bootstrap no aparece algun nivel)
  for (v in names(data_cat)) {
    data_bootstrap[[v]] <- factor(
      data_bootstrap[[v]],
      levels = levels(data_cat[[v]])
    )
  }
  
  # 3. Aprender una red bayesiana general (NO usamos blacklist)
  red_bootstrap <- hc(
    data_bootstrap,
    score = "bic",
    restart = 20,
    perturb = 5
    #maxp = 4
  )
  
  # 4. Extraer el Markov blanket de popularidad
  mb_b <- mb(
    red_bootstrap,
    node = variable_objetivo
  )
  
  # 5. Sumamos 1 al contador de las variables que aparecen en el MB
  contador_mb[mb_b] <- contador_mb[mb_b] + 1
  }

#### RESUMEN DE RESULTADOS

resumen_mb_bic <- data.frame(
  variable = names(contador_mb),
  apariciones = as.numeric(contador_mb),
  aparece = as.numeric(contador_mb) > 0
)

# orden por mayor aparición
resumen_mb_bic <- resumen_mb_bic[
  order(
    -resumen_mb_bic$apariciones,
    resumen_mb_bic$variable
  ),
]

cat("\n========================================\n")
cat("RESUMEN DE APARICIONES EN MARKOV BLANKET - BIC\n")
cat("========================================\n")
print(resumen_mb_bic)

# Variables seleccionadas: aparecen en algún MB
variables_seleccionadas_mb_bic <- names(contador_mb)[contador_mb > 0]

cat("\n========================================\n")
cat("VARIABLES SELECCIONADAS POR MARKOV BLANKET - BIC\n")
cat("========================================\n")
print(variables_seleccionadas_mb_bic)

# BASE DE DATOS REDUCIDA - BIC
############################################################################## 

vars_cat_mb_bic <- c(
  variable_objetivo,
  variables_seleccionadas_mb_bic
)

data_cat_mb_bic <- data_cat[,vars_cat_mb_bic]

data_cat_mb_bic <- droplevels(data_cat_mb_bic)

cat("\n========================================\n")
cat("BASE DE DATOS REDUCIDA - BIC\n")
cat("========================================\n")
str(data_cat_mb_bic)

############################################################################## 
# MARKOV BLANKET - AIC
##############################################################################

# Variable objetivo
variable_objetivo <- "popularidad"

# Variables candidatas, excluyendo la variable objetivo
variables_predictoras <- setdiff(
  names(data_cat),
  variable_objetivo
)

# Número de muestras bootstrap
R_bootstrap <- 100

# Contador: suma 1 cada vez que una variable aparece en el MB de popularidad
contador_mb <- rep(
  0,
  length(variables_predictoras)
)

names(contador_mb) <- variables_predictoras

#### BUCLE BOOTSTRAP

for (b in 1:R_bootstrap) {
  
  cat("\nRepetición bootstrap:", b, "de", R_bootstrap, "\n") # esto lo quitare pero de momento va bine para ver el proceso
  
  # 1. Generar muestra bootstrap de data_cat
  indices_bootstrap <- sample(
    1:nrow(data_cat),
    size = nrow(data_cat),
    replace = TRUE
  )
  
  data_bootstrap <- data_cat[indices_bootstrap,]
  
  # 2. Mantener los niveles originales de los factores (por si en alguna muetsra bootstrap no aparece algun nivel)
  for (v in names(data_cat)) {
    data_bootstrap[[v]] <- factor(
      data_bootstrap[[v]],
      levels = levels(data_cat[[v]])
    )
  }
  
  # 3. Aprender una red bayesiana general (NO usamos blacklist)
  red_bootstrap <- hc(
    data_bootstrap,
    score = "aic",
    restart = 20,
    perturb = 5
    #maxp = 4
  )
  
  # 4. Extraer el Markov blanket de popularidad
  mb_b <- mb(
    red_bootstrap,
    node = variable_objetivo
  )
  
  # 5. Sumamos 1 al contador de las variables que aparecen en el MB
  contador_mb[mb_b] <- contador_mb[mb_b] + 1
  }

#### RESUMEN DE RESULTADOS

resumen_mb_aic <- data.frame(
  variable = names(contador_mb),
  apariciones = as.numeric(contador_mb),
  aparece = as.numeric(contador_mb) > 0
)

# orden por mayor aparición
resumen_mb_aic <- resumen_mb_aic[
  order(
    -resumen_mb_aic$apariciones,
    resumen_mb_aic$variable
  ),
]

cat("\n========================================\n")
cat("RESUMEN DE APARICIONES EN MARKOV BLANKET - AIC\n")
cat("========================================\n")
print(resumen_mb_aic)

# Variables seleccionadas: aparecen en algún MB
variables_seleccionadas_mb_aic <- names(contador_mb)[contador_mb > 0]

cat("\n========================================\n")
cat("VARIABLES SELECCIONADAS POR MARKOV BLANKET - AIC\n")
cat("========================================\n")
print(variables_seleccionadas_mb_aic)

# BASE DE DATOS REDUCIDA - AIC
############################################################################## 

vars_cat_mb_aic <- c(
  variable_objetivo,
  variables_seleccionadas_mb_aic
)

data_cat_mb_aic <- data_cat[,vars_cat_mb_aic]

data_cat_mb_aic <- droplevels(data_cat_mb_aic)

cat("\n========================================\n")
cat("BASE DE DATOS REDUCIDA - AIC\n")
cat("========================================\n")
str(data_cat_mb_aic)


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# BASES DE DATOS FINALES
##############################################################################
datos_reducida <- data_cat_mb_bic
datos_completa <- data_cat # tambien podriamos usar data_cat_mb_aic


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
##############################################################################
# BASE DE DATOS COMPLETA
##############################################################################
##############################################################################



############################################################################## 
# ENTRENAMIENTO DE MODELOS (k-fold cross-validation + Ord-MAP/MAP)
##############################################################################

#### datos_completa

variable_objetivo <- "popularidad"

niveles <- c("muy_baja", "baja", "media", "alta")

datos_completa$popularidad <- factor(
  datos_completa$popularidad,
  levels = niveles # aseguramos orden de los niveles
)

#### CREAR FOLDS ESTRATIFICADOS

k <- 10

folds <- rep(NA, nrow(datos_completa))

for (clase in niveles) {
  
  indices_clase <- which(datos_completa$popularidad == clase)
  
  indices_clase <- sample(indices_clase)
  
  folds[indices_clase] <- rep(
    1:k,
    length.out = length(indices_clase)
  )
}

cat("PROPORCIÓN DE LOS FOLDS\n")

# Comprobamos que cada fold mantiene aproximadamente la proporción de clases
table(folds, datos_completa$popularidad)

round(prop.table(table(folds, datos_completa$popularidad), margin = 1), 3)

#### FUNCIÓN PARA ORD-MAP

OrdMAP <- function(x, levels) {
  
  probs_acumuladas <- cumsum(x)
  
  posicion <- which(probs_acumuladas >= 0.5)[1]
  
  return(levels[posicion])
}


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# NAIVE BAYES
##############################################################################

cat("\n========================================\n")
cat("MODELO 1: NAIVE BAYES\n")
cat("========================================\n")

# OBJETOS PARA GUARDAR RESULTADOS
##############################################################################

predicciones_NB_OrdMAP <- rep(NA, nrow(datos_completa))
predicciones_NB_MAP <- rep(NA, nrow(datos_completa))

metricas_NB_OrdMAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

metricas_NB_MAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

# ENTRENAR MODELO
##############################################################################

for (i in 1:k) {
  
  cat("\n========================================\n")
  cat("Fold", i, "de", k, "\n")
  cat("========================================\n")
  
  # Separar training y validation
  
  training.set <- datos_completa[folds != i, ]
  validation.set <- datos_completa[folds == i, ]
  
  # Naive Bayes
  
  xarxa <- naive.bayes(
    training.set,
    variable_objetivo
  )
  
  # Parámetros
  
  xarxa.estimada <- bn.fit(
    xarxa,
    training.set,
    method = "bayes",
    iss = 1 # usamos suavizado para controlar probabilidades 0
  )
  
  # Pasar a gRain
  
  xarxa.grain <- as.grain(xarxa.estimada)
  
  # Probabilidades posteriores
  
  pred <- predict(
    xarxa.grain,
    response = variable_objetivo,
    newdata = validation.set,
    type = "distribution" 
  )
  
  matriu_probs <- pred$pred[[variable_objetivo]]
  
  matriu_probs <- as.matrix(matriu_probs)
  
  matriu_probs <- matriu_probs[, niveles]
  
  # PREDICCIÓN ORD-MAP
  ##########################################################################
  
  predOM <- apply(
    matriu_probs,
    1,
    OrdMAP,
    levels = colnames(matriu_probs)
  )
  
  predOM <- factor(
    predOM,
    levels = niveles
  )
  
  # PREDICCIÓN MAP
  ##########################################################################
  
  predMAP <- colnames(matriu_probs)[
    max.col(matriu_probs, ties.method = "first")
  ]
  
  predMAP <- factor(
    predMAP,
    levels = niveles
  )
  
  # GUARDAR PREDICCIONES
  ##########################################################################
  
  predicciones_NB_OrdMAP[folds == i] <- as.character(predOM)
  predicciones_NB_MAP[folds == i] <- as.character(predMAP)
  
  # OBSERVACIONES REALES
  ##########################################################################
  
  obs <- validation.set$popularidad
  
  # MATRIZ DE CONFUSIÓN ORD-MAP
  ##########################################################################
  
  MC_OrdMAP <- table(
    Predicha = predOM,
    Observada = obs
  )
  
  cat("\nMatriz de confusión NB - Ord-MAP\n")
  print(MC_OrdMAP)
  
  # MÉTRICAS ORD-MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_OrdMAP)) / sum(MC_OrdMAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_OrdMAP) / colSums(MC_OrdMAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predOM)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas Ord-MAP
  
  metricas_NB_OrdMAP$AMAE[i] <- amae
  metricas_NB_OrdMAP$MAE[i] <- mae
  metricas_NB_OrdMAP$Accuracy[i] <- accuracy
  metricas_NB_OrdMAP$Balanced_Accuracy[i] <- balanced_accuracy
  
  # MATRIZ DE CONFUSIÓN MAP
  ##########################################################################
  
  MC_MAP <- table(
    Predicha = predMAP,
    Observada = obs
  )
  
  cat("\nMatriz de confusión NB - MAP\n")
  print(MC_MAP)
  
  # MÉTRICAS MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_MAP)) / sum(MC_MAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_MAP) / colSums(MC_MAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predMAP)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas MAP
  
  metricas_NB_MAP$AMAE[i] <- amae
  metricas_NB_MAP$MAE[i] <- mae
  metricas_NB_MAP$Accuracy[i] <- accuracy
  metricas_NB_MAP$Balanced_Accuracy[i] <- balanced_accuracy
}

# RESULTADOS POR FOLD - ORD-MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - NB ORD-MAP\n")
cat("========================================\n")

metricas_NB_OrdMAP

# RESULTADOS POR FOLD - MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - NB MAP\n")
cat("========================================\n")

metricas_NB_MAP

# RESUMEN DE MÉTRICAS - ORD-MAP
##############################################################################

resumen_metricas_NB_OrdMAP <- data.frame(
  criterio = "Ord-MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_NB_OrdMAP$AMAE),
    mean(metricas_NB_OrdMAP$MAE),
    mean(metricas_NB_OrdMAP$Accuracy),
    mean(metricas_NB_OrdMAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_NB_OrdMAP$AMAE),
    sd(metricas_NB_OrdMAP$MAE),
    sd(metricas_NB_OrdMAP$Accuracy),
    sd(metricas_NB_OrdMAP$Balanced_Accuracy)
  )
)

resumen_metricas_NB_OrdMAP

# RESUMEN DE MÉTRICAS - MAP
##############################################################################

resumen_metricas_NB_MAP <- data.frame(
  criterio = "MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_NB_MAP$AMAE),
    mean(metricas_NB_MAP$MAE),
    mean(metricas_NB_MAP$Accuracy),
    mean(metricas_NB_MAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_NB_MAP$AMAE),
    sd(metricas_NB_MAP$MAE),
    sd(metricas_NB_MAP$Accuracy),
    sd(metricas_NB_MAP$Balanced_Accuracy)
  )
)

resumen_metricas_NB_MAP

# COMPARACIÓN RESUMIDA ORD-MAP VS MAP
##############################################################################

comparacion_metricas_NB <- rbind(
  resumen_metricas_NB_OrdMAP,
  resumen_metricas_NB_MAP
)

comparacion_metricas_NB

# MATRIZ DE CONFUSIÓN GLOBAL - ORD-MAP
##############################################################################

predicciones_NB_OrdMAP <- factor(
  predicciones_NB_OrdMAP,
  levels = niveles
)

MC_global_NB_OrdMAP <- table(
  Predicha = predicciones_NB_OrdMAP,
  Observada = datos_completa$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - NB ORD-MAP\n")

MC_global_NB_OrdMAP

# MATRIZ DE CONFUSIÓN GLOBAL - MAP
##############################################################################

predicciones_NB_MAP <- factor(
  predicciones_NB_MAP,
  levels = niveles
)

MC_global_NB_MAP <- table(
  Predicha = predicciones_NB_MAP,
  Observada = datos_completa$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - NB MAP\n")

MC_global_NB_MAP

# MÉTRICAS DE COMPARACIÓN
##############################################################################

AMAE_NB_MAP <- metricas_NB_MAP$AMAE

AMAE_NB_ORDMAP <- metricas_NB_OrdMAP$AMAE


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# TREE-AUGMENTED NAIVE BAYES
##############################################################################

cat("\n========================================\n")
cat("MODELO 2: TREE-AUGMENTED NAIVE BAYES\n")
cat("========================================\n")

# OBJETOS PARA GUARDAR RESULTADOS
##############################################################################

predicciones_TAN_OrdMAP <- rep(NA, nrow(datos_completa))
predicciones_TAN_MAP <- rep(NA, nrow(datos_completa))

metricas_TAN_OrdMAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

metricas_TAN_MAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

# ENTRENAR MODELO
##############################################################################

for (i in 1:k) {
  
  cat("\n========================================\n")
  cat("Fold", i, "de", k, "\n")
  cat("========================================\n")
  
  # Separar training y validation
  
  training.set <- datos_completa[folds != i, ]
  validation.set <- datos_completa[folds == i, ]
  
  # Tree-Augmented Naive Bayes
  
  xarxa <- tree.bayes(
    training.set,
    variable_objetivo
  )
  
  # Parámetros
  
  xarxa.estimada <- bn.fit(
    xarxa,
    training.set,
    method = "bayes",
    iss = 1
  )
  
  # Pasar a gRain
  
  xarxa.grain <- as.grain(xarxa.estimada)
  
  # Probabilidades posteriores
  
  pred <- predict(
    xarxa.grain,
    response = variable_objetivo,
    newdata = validation.set,
    type = "distribution"
  )
  
  matriu_probs <- pred$pred[[variable_objetivo]]
  
  matriu_probs <- as.matrix(matriu_probs)
  
  matriu_probs <- matriu_probs[, niveles]
  
  # PREDICCIÓN ORD-MAP
  ##########################################################################
  
  predOM <- apply(
    matriu_probs,
    1,
    OrdMAP,
    levels = colnames(matriu_probs)
  )
  
  predOM <- factor(
    predOM,
    levels = niveles
  )
  
  # PREDICCIÓN MAP
  ##########################################################################
  
  predMAP <- colnames(matriu_probs)[
    max.col(matriu_probs, ties.method = "first")
  ]
  
  predMAP <- factor(
    predMAP,
    levels = niveles
  )
  
  # GUARDAR PREDICCIONES
  ##########################################################################
  
  predicciones_TAN_OrdMAP[folds == i] <- as.character(predOM)
  predicciones_TAN_MAP[folds == i] <- as.character(predMAP)
  
  # OBSERVACIONES REALES
  ##########################################################################
  
  obs <- validation.set$popularidad
  
  # MATRIZ DE CONFUSIÓN ORD-MAP
  ##########################################################################
  
  MC_OrdMAP <- table(
    Predicha = predOM,
    Observada = obs
  )
  
  cat("\nMatriz de confusión TAN - Ord-MAP\n")
  print(MC_OrdMAP)
  
  # MÉTRICAS ORD-MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_OrdMAP)) / sum(MC_OrdMAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_OrdMAP) / colSums(MC_OrdMAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predOM)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas Ord-MAP
  
  metricas_TAN_OrdMAP$AMAE[i] <- amae
  metricas_TAN_OrdMAP$MAE[i] <- mae
  metricas_TAN_OrdMAP$Accuracy[i] <- accuracy
  metricas_TAN_OrdMAP$Balanced_Accuracy[i] <- balanced_accuracy
  
  # MATRIZ DE CONFUSIÓN MAP
  ##########################################################################
  
  MC_MAP <- table(
    Predicha = predMAP,
    Observada = obs
  )
  
  cat("\nMatriz de confusión TAN - MAP\n")
  print(MC_MAP)
  
  # MÉTRICAS MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_MAP)) / sum(MC_MAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_MAP) / colSums(MC_MAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predMAP)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas MAP
  
  metricas_TAN_MAP$AMAE[i] <- amae
  metricas_TAN_MAP$MAE[i] <- mae
  metricas_TAN_MAP$Accuracy[i] <- accuracy
  metricas_TAN_MAP$Balanced_Accuracy[i] <- balanced_accuracy
}

# RESULTADOS POR FOLD - ORD-MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - TAN ORD-MAP\n")
cat("========================================\n")

metricas_TAN_OrdMAP

# RESULTADOS POR FOLD - MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - TAN MAP\n")
cat("========================================\n")

metricas_TAN_MAP

# RESUMEN DE MÉTRICAS - ORD-MAP
##############################################################################

resumen_metricas_TAN_OrdMAP <- data.frame(
  criterio = "Ord-MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_TAN_OrdMAP$AMAE),
    mean(metricas_TAN_OrdMAP$MAE),
    mean(metricas_TAN_OrdMAP$Accuracy),
    mean(metricas_TAN_OrdMAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_TAN_OrdMAP$AMAE),
    sd(metricas_TAN_OrdMAP$MAE),
    sd(metricas_TAN_OrdMAP$Accuracy),
    sd(metricas_TAN_OrdMAP$Balanced_Accuracy)
  )
)

resumen_metricas_TAN_OrdMAP

# RESUMEN DE MÉTRICAS - MAP
##############################################################################

resumen_metricas_TAN_MAP <- data.frame(
  criterio = "MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_TAN_MAP$AMAE),
    mean(metricas_TAN_MAP$MAE),
    mean(metricas_TAN_MAP$Accuracy),
    mean(metricas_TAN_MAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_TAN_MAP$AMAE),
    sd(metricas_TAN_MAP$MAE),
    sd(metricas_TAN_MAP$Accuracy),
    sd(metricas_TAN_MAP$Balanced_Accuracy)
  )
)

resumen_metricas_TAN_MAP

# COMPARACIÓN RESUMIDA ORD-MAP VS MAP
##############################################################################

comparacion_metricas_TAN <- rbind(
  resumen_metricas_TAN_OrdMAP,
  resumen_metricas_TAN_MAP
)

comparacion_metricas_TAN

# MATRIZ DE CONFUSIÓN GLOBAL - ORD-MAP
##############################################################################

predicciones_TAN_OrdMAP <- factor(
  predicciones_TAN_OrdMAP,
  levels = niveles
)

MC_global_TAN_OrdMAP <- table(
  Predicha = predicciones_TAN_OrdMAP,
  Observada = datos_completa$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - TAN ORD-MAP\n")

MC_global_TAN_OrdMAP

# MATRIZ DE CONFUSIÓN GLOBAL - MAP
##############################################################################

predicciones_TAN_MAP <- factor(
  predicciones_TAN_MAP,
  levels = niveles
)

MC_global_TAN_MAP <- table(
  Predicha = predicciones_TAN_MAP,
  Observada = datos_completa$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - TAN MAP\n")

MC_global_TAN_MAP

# MÉTRICAS DE COMPARACIÓN
##############################################################################
AMAE_TAN_MAP <- metricas_TAN_MAP$AMAE

AMAE_TAN_ORDMAP <- metricas_TAN_OrdMAP$AMAE


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# PROGNOSI
##############################################################################

cat("\n========================================\n")
cat("MODELO 3: PROGNOSI\n")
cat("========================================\n")

# OBJETOS PARA GUARDAR RESULTADOS
##############################################################################

predicciones_PROG_OrdMAP <- rep(NA, nrow(datos_completa))
predicciones_PROG_MAP <- rep(NA, nrow(datos_completa))

metricas_PROG_OrdMAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

metricas_PROG_MAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

# ENTRENAR MODELO
##############################################################################

for (i in 1:k) {
  
  cat("\n========================================\n")
  cat("Fold", i, "de", k, "\n")
  cat("========================================\n")
  
  # Separar training y validation
  
  training.set <- datos_completa[folds != i, ]
  validation.set <- datos_completa[folds == i, ]
  
  # Variables predictoras
  
  variables_predictoras <- setdiff(
    names(training.set),
    variable_objetivo
  )
  
  # Blacklist para modelo de prognosi
  # Prohibimos arcos desde popularidad hacia las variables predictoras
  
  blacklist_prognosi <- data.frame(
    from = rep(variable_objetivo, length(variables_predictoras)),
    to = variables_predictoras
  )
  
  # Aprender estructura de prognosi
  
  xarxa <- hc(
    training.set,
    score = "bic",
    blacklist = blacklist_prognosi,
    restart = 20,
    perturb = 5
  )
  
  # Parámetros
  
  xarxa.estimada <- bn.fit(
    xarxa,
    training.set,
    method = "bayes",
    iss = 1
  )
  
  # Pasar a gRain
  
  xarxa.grain <- as.grain(xarxa.estimada)
  
  # Probabilidades posteriores
  
  pred <- predict(
    xarxa.grain,
    response = variable_objetivo,
    newdata = validation.set,
    type = "distribution"
  )
  
  matriu_probs <- pred$pred[[variable_objetivo]]
  
  matriu_probs <- as.matrix(matriu_probs)
  
  matriu_probs <- matriu_probs[, niveles]
  
  # PREDICCIÓN ORD-MAP
  ##########################################################################
  
  predOM <- apply(
    matriu_probs,
    1,
    OrdMAP,
    levels = colnames(matriu_probs)
  )
  
  predOM <- factor(
    predOM,
    levels = niveles
  )
  
  # PREDICCIÓN MAP
  ##########################################################################
  
  predMAP <- colnames(matriu_probs)[
    max.col(matriu_probs, ties.method = "first")
  ]
  
  predMAP <- factor(
    predMAP,
    levels = niveles
  )
  
  # GUARDAR PREDICCIONES
  ##########################################################################
  
  predicciones_PROG_OrdMAP[folds == i] <- as.character(predOM)
  predicciones_PROG_MAP[folds == i] <- as.character(predMAP)
  
  # OBSERVACIONES REALES
  ##########################################################################
  
  obs <- validation.set$popularidad
  
  # MATRIZ DE CONFUSIÓN ORD-MAP
  ##########################################################################
  
  MC_OrdMAP <- table(
    Predicha = predOM,
    Observada = obs
  )
  
  cat("\nMatriz de confusión PROG - Ord-MAP\n")
  print(MC_OrdMAP)
  
  # MÉTRICAS ORD-MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_OrdMAP)) / sum(MC_OrdMAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_OrdMAP) / colSums(MC_OrdMAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predOM)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas Ord-MAP
  
  metricas_PROG_OrdMAP$AMAE[i] <- amae
  metricas_PROG_OrdMAP$MAE[i] <- mae
  metricas_PROG_OrdMAP$Accuracy[i] <- accuracy
  metricas_PROG_OrdMAP$Balanced_Accuracy[i] <- balanced_accuracy
  
  # MATRIZ DE CONFUSIÓN MAP
  ##########################################################################
  
  MC_MAP <- table(
    Predicha = predMAP,
    Observada = obs
  )
  
  cat("\nMatriz de confusión PROG - MAP\n")
  print(MC_MAP)
  
  # MÉTRICAS MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_MAP)) / sum(MC_MAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_MAP) / colSums(MC_MAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predMAP)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas MAP
  
  metricas_PROG_MAP$AMAE[i] <- amae
  metricas_PROG_MAP$MAE[i] <- mae
  metricas_PROG_MAP$Accuracy[i] <- accuracy
  metricas_PROG_MAP$Balanced_Accuracy[i] <- balanced_accuracy
}

# RESULTADOS POR FOLD - ORD-MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - PROG ORD-MAP\n")
cat("========================================\n")

metricas_PROG_OrdMAP

# RESULTADOS POR FOLD - MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - PROG MAP\n")
cat("========================================\n")

metricas_PROG_MAP

# RESUMEN DE MÉTRICAS - ORD-MAP
##############################################################################

resumen_metricas_PROG_OrdMAP <- data.frame(
  criterio = "Ord-MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_PROG_OrdMAP$AMAE),
    mean(metricas_PROG_OrdMAP$MAE),
    mean(metricas_PROG_OrdMAP$Accuracy),
    mean(metricas_PROG_OrdMAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_PROG_OrdMAP$AMAE),
    sd(metricas_PROG_OrdMAP$MAE),
    sd(metricas_PROG_OrdMAP$Accuracy),
    sd(metricas_PROG_OrdMAP$Balanced_Accuracy)
  )
)

resumen_metricas_PROG_OrdMAP

# RESUMEN DE MÉTRICAS - MAP
##############################################################################

resumen_metricas_PROG_MAP <- data.frame(
  criterio = "MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_PROG_MAP$AMAE),
    mean(metricas_PROG_MAP$MAE),
    mean(metricas_PROG_MAP$Accuracy),
    mean(metricas_PROG_MAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_PROG_MAP$AMAE),
    sd(metricas_PROG_MAP$MAE),
    sd(metricas_PROG_MAP$Accuracy),
    sd(metricas_PROG_MAP$Balanced_Accuracy)
  )
)

resumen_metricas_PROG_MAP

# COMPARACIÓN RESUMIDA ORD-MAP VS MAP
##############################################################################

comparacion_metricas_PROG <- rbind(
  resumen_metricas_PROG_OrdMAP,
  resumen_metricas_PROG_MAP
)

comparacion_metricas_PROG

# MATRIZ DE CONFUSIÓN GLOBAL - ORD-MAP
##############################################################################

predicciones_PROG_OrdMAP <- factor(
  predicciones_PROG_OrdMAP,
  levels = niveles
)

MC_global_PROG_OrdMAP <- table(
  Predicha = predicciones_PROG_OrdMAP,
  Observada = datos_completa$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - PROG ORD-MAP\n")

MC_global_PROG_OrdMAP

# MATRIZ DE CONFUSIÓN GLOBAL - MAP
##############################################################################

predicciones_PROG_MAP <- factor(
  predicciones_PROG_MAP,
  levels = niveles
)

MC_global_PROG_MAP <- table(
  Predicha = predicciones_PROG_MAP,
  Observada = datos_completa$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - PROG MAP\n")

MC_global_PROG_MAP

# MÉTRICAS DE COMPARACIÓN
##############################################################################

AMAE_PROG_MAP <- metricas_PROG_MAP$AMAE

AMAE_PROG_ORDMAP <- metricas_PROG_OrdMAP$AMAE


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# TABLA DE RESULTADOS - Base completa (para poder mostrar en el informe)
##############################################################################

comparacion_metricas_NB$modelo <- "NB"
comparacion_metricas_TAN$modelo <- "TAN"
comparacion_metricas_PROG$modelo <- "PROG"

resultados_base_completa <- rbind(
  comparacion_metricas_NB,
  comparacion_metricas_TAN,
  comparacion_metricas_PROG
)

resultados_base_completa <- resultados_base_completa[
  ,
  c("modelo", "criterio", "metrica", "media", "desviacion_tipica")
] # para rdenar y que "modelo" salaga primero

rownames(resultados_base_completa) <- NULL

cat("\n========================================\n")
cat("RESULTADOS BASE COMPLETA\n")
cat("========================================\n")

resultados_base_completa

##############################################################################
# MATRICES DE COFUSIÓN GLOBALES - Base completa (para poder mostrar en el informe)
##############################################################################

MC_global_base_completa <- list(
  NB_OrdMAP = MC_global_NB_OrdMAP,
  NB_MAP = MC_global_NB_MAP,
  TAN_OrdMAP = MC_global_TAN_OrdMAP,
  TAN_MAP = MC_global_TAN_MAP,
  PROG_OrdMAP = MC_global_PROG_OrdMAP,
  PROG_MAP = MC_global_PROG_MAP
)


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# COMPARACIÓN DE MODELOS - Base completa
##############################################################################

AMAE_comparacion <- data.frame(
  NB_MAP = AMAE_NB_MAP,
  NB_ORDMAP = AMAE_NB_ORDMAP,
  TAN_MAP = AMAE_TAN_MAP,
  TAN_ORDMAP = AMAE_TAN_ORDMAP,
  PROG_MAP = AMAE_PROG_MAP,
  PROG_ORDMAP = AMAE_PROG_ORDMAP
)

AMAE_comparacion

# FRIEDMAN (para descartar igualdad global)
##############################################################################

friedman_completa <- friedman.test(
  as.matrix(AMAE_comparacion)
) 

friedman_completa # p-value = 1.47e-08 --> hay diferencias --> tests dos a dos

# RANKING SEGÚN AMAE MEDIO
##############################################################################

ranking_AMAE <- data.frame(
  metodo = colnames(AMAE_comparacion),
  AMAE_medio = colMeans(AMAE_comparacion),
  desviacion_tipica = apply(AMAE_comparacion, 2, sd)
)

ranking_AMAE <- ranking_AMAE[
  order(ranking_AMAE$AMAE_medio),
]

ranking_AMAE_completa <- ranking_AMAE # para poder mostrar en informe (porque se sobreescribe)

cat("\n========================================\n")
cat("RANKING AMAE - Base completa\n")
cat("========================================\n")

ranking_AMAE_completa

# el mejor modelo es "NB_ORDMAP"

# COMPARACIÓN UNILATERAL DEL MEJOR MODELO CONTRA EL RESTO
##############################################################################

mejor_modelo <- ranking_AMAE[[1,1]]

modelos_restantes <- setdiff(
  colnames(AMAE_comparacion),
  mejor_modelo
)

resultado_unilateral <- data.frame(
  modelo_referencia = mejor_modelo,
  modelo_comparado = modelos_restantes,
  media_referencia = NA,
  media_comparado = NA,
  p_shapiro = NA,
  normalidad = NA,
  test_aplicado = NA,
  p_valor = NA
)

for (j in 1:length(modelos_restantes)) {
  
  modelo_comparado <- modelos_restantes[j]
  
  x <- AMAE_comparacion[[mejor_modelo]]
  y <- AMAE_comparacion[[modelo_comparado]]
  
  diferencias <- x - y
  
  resultado_unilateral$media_referencia[j] <- mean(x)
  resultado_unilateral$media_comparado[j] <- mean(y)
  
  # Shapiro-Wilk sobre diferencias (para testear Normalidad)
  test_shapiro <- shapiro.test(diferencias)
  
  resultado_unilateral$p_shapiro[j] <- test_shapiro$p.value
  
  # Queremos contrastar:
  # H1: AMAE_NB_ORDMAP < AMAE_modelo_comparado
  
  # si Normalidad --> t-test
  if (test_shapiro$p.value >= 0.05) {
    
    resultado_unilateral$normalidad[j] <- "Normalidad"
    
    test_diferencias <- t.test(
      x,
      y,
      paired = TRUE,
      alternative = "less"
    )
    
    resultado_unilateral$test_aplicado[j] <- "t-test apareado unilateral"
    resultado_unilateral$p_valor[j] <- test_diferencias$p.value
  }
  
  # si NO Normalidad --> Wilcoxon
  
  if (test_shapiro$p.value < 0.05) {
    
    resultado_unilateral$normalidad[j] <- "No normalidad"
    
    test_diferencias <- wilcox.test(
      x,
      y,
      paired = TRUE,
      alternative = "less",
      exact = FALSE
    )
    
    resultado_unilateral$test_aplicado[j] <- "Wilcoxon apareado unilateral"
    resultado_unilateral$p_valor[j] <- test_diferencias$p.value
  }
}

# Corrección por comparaciones múltiples
resultado_unilateral$p_valor_ajustado_Holm <- p.adjust(
  resultado_unilateral$p_valor,
  method = "holm"
)

resultado_unilateral$significativo <- resultado_unilateral$p_valor_ajustado_Holm < 0.05

resultado_unilateral_completa <- resultado_unilateral # para poder mostrar en informe porque se sobreescribe

cat("\n========================================\n")
cat("COMPARACIÓN - Base completa\n")
cat("========================================\n")

resultado_unilateral_completa


## -------------------------------------------------------------------------------------------------------------
#################### BASE COMPLETA: seleccionamos el modelo Naive Bayes con criterio Ord-MAP, por parsimonia ####################################

AMAE_completa <- AMAE_comparacion[["NB_ORDMAP"]]


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
##############################################################################
# BASE DE DATOS REDUCIDA
##############################################################################
##############################################################################



############################################################################## 
# ENTRENAMIENTO DE MODELOS (k-fold cross-validation + Ord-MAP/MAP)
##############################################################################

#### datos_reducida

variable_objetivo <- "popularidad"

niveles <- c("muy_baja", "baja", "media", "alta")

datos_reducida$popularidad <- factor(
  datos_reducida$popularidad,
  levels = niveles # aseguramos orden de los niveles
)

#### CREAR FOLDS ESTRATIFICADOS

# utilizamos los mismo que antes para poder comparar apareadamente entre datos_completa y datos_reducida

#### FUNCIÓN PARA ORD-MAP

# ya está creada


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# NAIVE BAYES
##############################################################################

cat("\n========================================\n")
cat("MODELO 1: NAIVE BAYES\n")
cat("========================================\n")

# OBJETOS PARA GUARDAR RESULTADOS
##############################################################################

predicciones_NB_OrdMAP <- rep(NA, nrow(datos_reducida))
predicciones_NB_MAP <- rep(NA, nrow(datos_reducida))

metricas_NB_OrdMAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

metricas_NB_MAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

# ENTRENAR MODELO
##############################################################################

for (i in 1:k) {
  
  cat("\n========================================\n")
  cat("Fold", i, "de", k, "\n")
  cat("========================================\n")
  
  # Separar training y validation
  
  training.set <- datos_reducida[folds != i, ]
  validation.set <- datos_reducida[folds == i, ]
  
  # Naive Bayes
  
  xarxa <- naive.bayes(
    training.set,
    variable_objetivo
  )
  
  # Parámetros
  
  xarxa.estimada <- bn.fit(
    xarxa,
    training.set,
    method = "bayes",
    iss = 1 # usamos suavizado para controlar probabilidades 0
  )
  
  # Pasar a gRain
  
  xarxa.grain <- as.grain(xarxa.estimada)
  
  # Probabilidades posteriores
  
  pred <- predict(
    xarxa.grain,
    response = variable_objetivo,
    newdata = validation.set,
    type = "distribution" 
  )
  
  matriu_probs <- pred$pred[[variable_objetivo]]
  
  matriu_probs <- as.matrix(matriu_probs)
  
  matriu_probs <- matriu_probs[, niveles]
  
  # PREDICCIÓN ORD-MAP
  ##########################################################################
  
  predOM <- apply(
    matriu_probs,
    1,
    OrdMAP,
    levels = colnames(matriu_probs)
  )
  
  predOM <- factor(
    predOM,
    levels = niveles
  )
  
  # PREDICCIÓN MAP
  ##########################################################################
  
  predMAP <- colnames(matriu_probs)[
    max.col(matriu_probs, ties.method = "first")
  ]
  
  predMAP <- factor(
    predMAP,
    levels = niveles
  )
  
  # GUARDAR PREDICCIONES
  ##########################################################################
  
  predicciones_NB_OrdMAP[folds == i] <- as.character(predOM)
  predicciones_NB_MAP[folds == i] <- as.character(predMAP)
  
  # OBSERVACIONES REALES
  ##########################################################################
  
  obs <- validation.set$popularidad
  
  # MATRIZ DE CONFUSIÓN ORD-MAP
  ##########################################################################
  
  MC_OrdMAP <- table(
    Predicha = predOM,
    Observada = obs
  )
  
  cat("\nMatriz de confusión NB - Ord-MAP\n")
  print(MC_OrdMAP)
  
  # MÉTRICAS ORD-MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_OrdMAP)) / sum(MC_OrdMAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_OrdMAP) / colSums(MC_OrdMAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predOM)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas Ord-MAP
  
  metricas_NB_OrdMAP$AMAE[i] <- amae
  metricas_NB_OrdMAP$MAE[i] <- mae
  metricas_NB_OrdMAP$Accuracy[i] <- accuracy
  metricas_NB_OrdMAP$Balanced_Accuracy[i] <- balanced_accuracy
  
  # MATRIZ DE CONFUSIÓN MAP
  ##########################################################################
  
  MC_MAP <- table(
    Predicha = predMAP,
    Observada = obs
  )
  
  cat("\nMatriz de confusión NB - MAP\n")
  print(MC_MAP)
  
  # MÉTRICAS MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_MAP)) / sum(MC_MAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_MAP) / colSums(MC_MAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predMAP)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas MAP
  
  metricas_NB_MAP$AMAE[i] <- amae
  metricas_NB_MAP$MAE[i] <- mae
  metricas_NB_MAP$Accuracy[i] <- accuracy
  metricas_NB_MAP$Balanced_Accuracy[i] <- balanced_accuracy
}

# RESULTADOS POR FOLD - ORD-MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - NB ORD-MAP\n")
cat("========================================\n")

metricas_NB_OrdMAP

# RESULTADOS POR FOLD - MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - NB MAP\n")
cat("========================================\n")

metricas_NB_MAP

# RESUMEN DE MÉTRICAS - ORD-MAP
##############################################################################

resumen_metricas_NB_OrdMAP <- data.frame(
  criterio = "Ord-MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_NB_OrdMAP$AMAE),
    mean(metricas_NB_OrdMAP$MAE),
    mean(metricas_NB_OrdMAP$Accuracy),
    mean(metricas_NB_OrdMAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_NB_OrdMAP$AMAE),
    sd(metricas_NB_OrdMAP$MAE),
    sd(metricas_NB_OrdMAP$Accuracy),
    sd(metricas_NB_OrdMAP$Balanced_Accuracy)
  )
)

resumen_metricas_NB_OrdMAP

# RESUMEN DE MÉTRICAS - MAP
##############################################################################

resumen_metricas_NB_MAP <- data.frame(
  criterio = "MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_NB_MAP$AMAE),
    mean(metricas_NB_MAP$MAE),
    mean(metricas_NB_MAP$Accuracy),
    mean(metricas_NB_MAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_NB_MAP$AMAE),
    sd(metricas_NB_MAP$MAE),
    sd(metricas_NB_MAP$Accuracy),
    sd(metricas_NB_MAP$Balanced_Accuracy)
  )
)

resumen_metricas_NB_MAP

# COMPARACIÓN RESUMIDA ORD-MAP VS MAP
##############################################################################

comparacion_metricas_NB <- rbind(
  resumen_metricas_NB_OrdMAP,
  resumen_metricas_NB_MAP
)

comparacion_metricas_NB

# MATRIZ DE CONFUSIÓN GLOBAL - ORD-MAP
##############################################################################

predicciones_NB_OrdMAP <- factor(
  predicciones_NB_OrdMAP,
  levels = niveles
)

MC_global_NB_OrdMAP <- table(
  Predicha = predicciones_NB_OrdMAP,
  Observada = datos_reducida$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - NB ORD-MAP\n")

MC_global_NB_OrdMAP

# MATRIZ DE CONFUSIÓN GLOBAL - MAP
##############################################################################

predicciones_NB_MAP <- factor(
  predicciones_NB_MAP,
  levels = niveles
)

MC_global_NB_MAP <- table(
  Predicha = predicciones_NB_MAP,
  Observada = datos_reducida$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - NB MAP\n")

MC_global_NB_MAP

# MÉTRICAS DE COMPARACIÓN
##############################################################################

AMAE_NB_MAP <- metricas_NB_MAP$AMAE

AMAE_NB_ORDMAP <- metricas_NB_OrdMAP$AMAE


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# TREE-AUGMENTED NAIVE BAYES
##############################################################################

cat("\n========================================\n")
cat("MODELO 2: TREE-AUGMENTED NAIVE BAYES\n")
cat("========================================\n")

# OBJETOS PARA GUARDAR RESULTADOS
##############################################################################

predicciones_TAN_OrdMAP <- rep(NA, nrow(datos_reducida))
predicciones_TAN_MAP <- rep(NA, nrow(datos_reducida))

metricas_TAN_OrdMAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

metricas_TAN_MAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

# ENTRENAR MODELO
##############################################################################

for (i in 1:k) {
  
  cat("\n========================================\n")
  cat("Fold", i, "de", k, "\n")
  cat("========================================\n")
  
  # Separar training y validation
  
  training.set <- datos_reducida[folds != i, ]
  validation.set <- datos_reducida[folds == i, ]
  
  # Tree-Augmented Naive Bayes
  
  xarxa <- tree.bayes(
    training.set,
    variable_objetivo
  )
  
  # Parámetros
  
  xarxa.estimada <- bn.fit(
    xarxa,
    training.set,
    method = "bayes",
    iss = 1
  )
  
  # Pasar a gRain
  
  xarxa.grain <- as.grain(xarxa.estimada)
  
  # Probabilidades posteriores
  
  pred <- predict(
    xarxa.grain,
    response = variable_objetivo,
    newdata = validation.set,
    type = "distribution"
  )
  
  matriu_probs <- pred$pred[[variable_objetivo]]
  
  matriu_probs <- as.matrix(matriu_probs)
  
  matriu_probs <- matriu_probs[, niveles]
  
  # PREDICCIÓN ORD-MAP
  ##########################################################################
  
  predOM <- apply(
    matriu_probs,
    1,
    OrdMAP,
    levels = colnames(matriu_probs)
  )
  
  predOM <- factor(
    predOM,
    levels = niveles
  )
  
  # PREDICCIÓN MAP
  ##########################################################################
  
  predMAP <- colnames(matriu_probs)[
    max.col(matriu_probs, ties.method = "first")
  ]
  
  predMAP <- factor(
    predMAP,
    levels = niveles
  )
  
  # GUARDAR PREDICCIONES
  ##########################################################################
  
  predicciones_TAN_OrdMAP[folds == i] <- as.character(predOM)
  predicciones_TAN_MAP[folds == i] <- as.character(predMAP)
  
  # OBSERVACIONES REALES
  ##########################################################################
  
  obs <- validation.set$popularidad
  
  # MATRIZ DE CONFUSIÓN ORD-MAP
  ##########################################################################
  
  MC_OrdMAP <- table(
    Predicha = predOM,
    Observada = obs
  )
  
  cat("\nMatriz de confusión TAN - Ord-MAP\n")
  print(MC_OrdMAP)
  
  # MÉTRICAS ORD-MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_OrdMAP)) / sum(MC_OrdMAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_OrdMAP) / colSums(MC_OrdMAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predOM)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas Ord-MAP
  
  metricas_TAN_OrdMAP$AMAE[i] <- amae
  metricas_TAN_OrdMAP$MAE[i] <- mae
  metricas_TAN_OrdMAP$Accuracy[i] <- accuracy
  metricas_TAN_OrdMAP$Balanced_Accuracy[i] <- balanced_accuracy
  
  # MATRIZ DE CONFUSIÓN MAP
  ##########################################################################
  
  MC_MAP <- table(
    Predicha = predMAP,
    Observada = obs
  )
  
  cat("\nMatriz de confusión TAN - MAP\n")
  print(MC_MAP)
  
  # MÉTRICAS MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_MAP)) / sum(MC_MAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_MAP) / colSums(MC_MAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predMAP)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas MAP
  
  metricas_TAN_MAP$AMAE[i] <- amae
  metricas_TAN_MAP$MAE[i] <- mae
  metricas_TAN_MAP$Accuracy[i] <- accuracy
  metricas_TAN_MAP$Balanced_Accuracy[i] <- balanced_accuracy
}

# RESULTADOS POR FOLD - ORD-MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - TAN ORD-MAP\n")
cat("========================================\n")

metricas_TAN_OrdMAP

# RESULTADOS POR FOLD - MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - TAN MAP\n")
cat("========================================\n")

metricas_TAN_MAP

# RESUMEN DE MÉTRICAS - ORD-MAP
##############################################################################

resumen_metricas_TAN_OrdMAP <- data.frame(
  criterio = "Ord-MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_TAN_OrdMAP$AMAE),
    mean(metricas_TAN_OrdMAP$MAE),
    mean(metricas_TAN_OrdMAP$Accuracy),
    mean(metricas_TAN_OrdMAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_TAN_OrdMAP$AMAE),
    sd(metricas_TAN_OrdMAP$MAE),
    sd(metricas_TAN_OrdMAP$Accuracy),
    sd(metricas_TAN_OrdMAP$Balanced_Accuracy)
  )
)

resumen_metricas_TAN_OrdMAP

# RESUMEN DE MÉTRICAS - MAP
##############################################################################

resumen_metricas_TAN_MAP <- data.frame(
  criterio = "MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_TAN_MAP$AMAE),
    mean(metricas_TAN_MAP$MAE),
    mean(metricas_TAN_MAP$Accuracy),
    mean(metricas_TAN_MAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_TAN_MAP$AMAE),
    sd(metricas_TAN_MAP$MAE),
    sd(metricas_TAN_MAP$Accuracy),
    sd(metricas_TAN_MAP$Balanced_Accuracy)
  )
)

resumen_metricas_TAN_MAP

# COMPARACIÓN RESUMIDA ORD-MAP VS MAP
##############################################################################

comparacion_metricas_TAN <- rbind(
  resumen_metricas_TAN_OrdMAP,
  resumen_metricas_TAN_MAP
)

comparacion_metricas_TAN

# MATRIZ DE CONFUSIÓN GLOBAL - ORD-MAP
##############################################################################

predicciones_TAN_OrdMAP <- factor(
  predicciones_TAN_OrdMAP,
  levels = niveles
)

MC_global_TAN_OrdMAP <- table(
  Predicha = predicciones_TAN_OrdMAP,
  Observada = datos_reducida$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - TAN ORD-MAP\n")

MC_global_TAN_OrdMAP

# MATRIZ DE CONFUSIÓN GLOBAL - MAP
##############################################################################

predicciones_TAN_MAP <- factor(
  predicciones_TAN_MAP,
  levels = niveles
)

MC_global_TAN_MAP <- table(
  Predicha = predicciones_TAN_MAP,
  Observada = datos_reducida$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - TAN MAP\n")

MC_global_TAN_MAP

# MÉTRICAS DE COMPARACIÓN
##############################################################################
AMAE_TAN_MAP <- metricas_TAN_MAP$AMAE

AMAE_TAN_ORDMAP <- metricas_TAN_OrdMAP$AMAE


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# PROGNOSI
##############################################################################

cat("\n========================================\n")
cat("MODELO 3: PROGNOSI\n")
cat("========================================\n")

# OBJETOS PARA GUARDAR RESULTADOS
##############################################################################

predicciones_PROG_OrdMAP <- rep(NA, nrow(datos_reducida))
predicciones_PROG_MAP <- rep(NA, nrow(datos_reducida))

metricas_PROG_OrdMAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

metricas_PROG_MAP <- data.frame(
  fold = 1:k,
  AMAE = NA,
  MAE = NA,
  Accuracy = NA,
  Balanced_Accuracy = NA
)

# ENTRENAR MODELO
##############################################################################

for (i in 1:k) {
  
  cat("\n========================================\n")
  cat("Fold", i, "de", k, "\n")
  cat("========================================\n")
  
  # Separar training y validation
  
  training.set <- datos_reducida[folds != i, ]
  validation.set <- datos_reducida[folds == i, ]
  
  # Variables predictoras
  
  variables_predictoras <- setdiff(
    names(training.set),
    variable_objetivo
  )
  
  # Blacklist para modelo de prognosi
  # Prohibimos arcos desde popularidad hacia las variables predictoras
  
  blacklist_prognosi <- data.frame(
    from = rep(variable_objetivo, length(variables_predictoras)),
    to = variables_predictoras
  )
  
  # Aprender estructura de prognosi
  
  xarxa <- hc(
    training.set,
    score = "bic",
    blacklist = blacklist_prognosi,
    restart = 20,
    perturb = 5
  )
  
  # Parámetros
  
  xarxa.estimada <- bn.fit(
    xarxa,
    training.set,
    method = "bayes",
    iss = 1
  )
  
  # Pasar a gRain
  
  xarxa.grain <- as.grain(xarxa.estimada)
  
  # Probabilidades posteriores
  
  pred <- predict(
    xarxa.grain,
    response = variable_objetivo,
    newdata = validation.set,
    type = "distribution"
  )
  
  matriu_probs <- pred$pred[[variable_objetivo]]
  
  matriu_probs <- as.matrix(matriu_probs)
  
  matriu_probs <- matriu_probs[, niveles]
  
  # PREDICCIÓN ORD-MAP
  ##########################################################################
  
  predOM <- apply(
    matriu_probs,
    1,
    OrdMAP,
    levels = colnames(matriu_probs)
  )
  
  predOM <- factor(
    predOM,
    levels = niveles
  )
  
  # PREDICCIÓN MAP
  ##########################################################################
  
  predMAP <- colnames(matriu_probs)[
    max.col(matriu_probs, ties.method = "first")
  ]
  
  predMAP <- factor(
    predMAP,
    levels = niveles
  )
  
  # GUARDAR PREDICCIONES
  ##########################################################################
  
  predicciones_PROG_OrdMAP[folds == i] <- as.character(predOM)
  predicciones_PROG_MAP[folds == i] <- as.character(predMAP)
  
  # OBSERVACIONES REALES
  ##########################################################################
  
  obs <- validation.set$popularidad
  
  # MATRIZ DE CONFUSIÓN ORD-MAP
  ##########################################################################
  
  MC_OrdMAP <- table(
    Predicha = predOM,
    Observada = obs
  )
  
  cat("\nMatriz de confusión PROG - Ord-MAP\n")
  print(MC_OrdMAP)
  
  # MÉTRICAS ORD-MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_OrdMAP)) / sum(MC_OrdMAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_OrdMAP) / colSums(MC_OrdMAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predOM)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas Ord-MAP
  
  metricas_PROG_OrdMAP$AMAE[i] <- amae
  metricas_PROG_OrdMAP$MAE[i] <- mae
  metricas_PROG_OrdMAP$Accuracy[i] <- accuracy
  metricas_PROG_OrdMAP$Balanced_Accuracy[i] <- balanced_accuracy
  
  # MATRIZ DE CONFUSIÓN MAP
  ##########################################################################
  
  MC_MAP <- table(
    Predicha = predMAP,
    Observada = obs
  )
  
  cat("\nMatriz de confusión PROG - MAP\n")
  print(MC_MAP)
  
  # MÉTRICAS MAP
  ##########################################################################
  
  # Accuracy
  
  accuracy <- sum(diag(MC_MAP)) / sum(MC_MAP)
  
  # Balanced Accuracy
  
  recall <- diag(MC_MAP) / colSums(MC_MAP)
  balanced_accuracy <- mean(recall, na.rm = TRUE)
  
  # MAE
  
  obs_num <- as.numeric(obs)
  pred_num <- as.numeric(predMAP)
  
  errores <- abs(pred_num - obs_num)
  
  mae <- mean(errores)
  
  # AMAE
  
  mae_clases <- rep(NA, length(niveles))
  names(mae_clases) <- niveles
  
  for (clase in niveles) {
    mae_clases[clase] <- mean(errores[obs == clase])
  }
  
  amae <- mean(mae_clases, na.rm = TRUE)
  
  # Guardar métricas MAP
  
  metricas_PROG_MAP$AMAE[i] <- amae
  metricas_PROG_MAP$MAE[i] <- mae
  metricas_PROG_MAP$Accuracy[i] <- accuracy
  metricas_PROG_MAP$Balanced_Accuracy[i] <- balanced_accuracy
}

# RESULTADOS POR FOLD - ORD-MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - PROG ORD-MAP\n")
cat("========================================\n")

metricas_PROG_OrdMAP

# RESULTADOS POR FOLD - MAP
##############################################################################

cat("\n========================================\n")
cat("MÉTRICAS POR FOLD - PROG MAP\n")
cat("========================================\n")

metricas_PROG_MAP

# RESUMEN DE MÉTRICAS - ORD-MAP
##############################################################################

resumen_metricas_PROG_OrdMAP <- data.frame(
  criterio = "Ord-MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_PROG_OrdMAP$AMAE),
    mean(metricas_PROG_OrdMAP$MAE),
    mean(metricas_PROG_OrdMAP$Accuracy),
    mean(metricas_PROG_OrdMAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_PROG_OrdMAP$AMAE),
    sd(metricas_PROG_OrdMAP$MAE),
    sd(metricas_PROG_OrdMAP$Accuracy),
    sd(metricas_PROG_OrdMAP$Balanced_Accuracy)
  )
)

resumen_metricas_PROG_OrdMAP

# RESUMEN DE MÉTRICAS - MAP
##############################################################################

resumen_metricas_PROG_MAP <- data.frame(
  criterio = "MAP",
  metrica = c("AMAE", "MAE", "Accuracy", "Balanced_Accuracy"),
  media = c(
    mean(metricas_PROG_MAP$AMAE),
    mean(metricas_PROG_MAP$MAE),
    mean(metricas_PROG_MAP$Accuracy),
    mean(metricas_PROG_MAP$Balanced_Accuracy)
  ),
  desviacion_tipica = c(
    sd(metricas_PROG_MAP$AMAE),
    sd(metricas_PROG_MAP$MAE),
    sd(metricas_PROG_MAP$Accuracy),
    sd(metricas_PROG_MAP$Balanced_Accuracy)
  )
)

resumen_metricas_PROG_MAP

# COMPARACIÓN RESUMIDA ORD-MAP VS MAP
##############################################################################

comparacion_metricas_PROG <- rbind(
  resumen_metricas_PROG_OrdMAP,
  resumen_metricas_PROG_MAP
)

comparacion_metricas_PROG

# MATRIZ DE CONFUSIÓN GLOBAL - ORD-MAP
##############################################################################

predicciones_PROG_OrdMAP <- factor(
  predicciones_PROG_OrdMAP,
  levels = niveles
)

MC_global_PROG_OrdMAP <- table(
  Predicha = predicciones_PROG_OrdMAP,
  Observada = datos_reducida$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - PROG ORD-MAP\n")

MC_global_PROG_OrdMAP

# MATRIZ DE CONFUSIÓN GLOBAL - MAP
##############################################################################

predicciones_PROG_MAP <- factor(
  predicciones_PROG_MAP,
  levels = niveles
)

MC_global_PROG_MAP <- table(
  Predicha = predicciones_PROG_MAP,
  Observada = datos_reducida$popularidad
)

cat("\nMATRIZ DE CONFUSIÓN GLOBAL - PROG MAP\n")

MC_global_PROG_MAP

# MÉTRICAS DE COMPARACIÓN
##############################################################################

AMAE_PROG_MAP <- metricas_PROG_MAP$AMAE

AMAE_PROG_ORDMAP <- metricas_PROG_OrdMAP$AMAE


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# TABLA DE RESULTADOS - Base reducida (para poder mostrar en el informe)
##############################################################################

comparacion_metricas_NB$modelo <- "NB"
comparacion_metricas_TAN$modelo <- "TAN"
comparacion_metricas_PROG$modelo <- "PROG"

resultados_base_reducida <- rbind(
  comparacion_metricas_NB,
  comparacion_metricas_TAN,
  comparacion_metricas_PROG
)

resultados_base_reducida <- resultados_base_reducida[
  ,
  c("modelo", "criterio", "metrica", "media", "desviacion_tipica")
]

rownames(resultados_base_reducida) <- NULL

cat("\n========================================\n")
cat("RESULTADOS BASE REDUCIDA\n")
cat("========================================\n")

resultados_base_reducida

##############################################################################
# MATRICES DE CONFUSIÓN GLOBALES - Base reducida (para poder mostrar en el informe)
##############################################################################

MC_global_base_reducida <- list(
  NB_OrdMAP = MC_global_NB_OrdMAP,
  NB_MAP = MC_global_NB_MAP,
  TAN_OrdMAP = MC_global_TAN_OrdMAP,
  TAN_MAP = MC_global_TAN_MAP,
  PROG_OrdMAP = MC_global_PROG_OrdMAP,
  PROG_MAP = MC_global_PROG_MAP
)


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# COMPARACIÓN DE MODELOS - Base reducida
##############################################################################

AMAE_comparacion <- data.frame(
  NB_MAP = AMAE_NB_MAP,
  NB_ORDMAP = AMAE_NB_ORDMAP,
  TAN_MAP = AMAE_TAN_MAP,
  TAN_ORDMAP = AMAE_TAN_ORDMAP,
  PROG_MAP = AMAE_PROG_MAP,
  PROG_ORDMAP = AMAE_PROG_ORDMAP
)

AMAE_comparacion

# FRIEDMAN (para descartar igualdad global)
##############################################################################

friedman_reducida <- friedman.test(
  as.matrix(AMAE_comparacion)
) 

friedman_reducida # p-value = 1.47e-08 --> hay diferencias --> tests dos a dos

# RANKING SEGÚN AMAE MEDIO
##############################################################################

ranking_AMAE <- data.frame(
  metodo = colnames(AMAE_comparacion),
  AMAE_medio = colMeans(AMAE_comparacion),
  desviacion_tipica = apply(AMAE_comparacion, 2, sd)
)

ranking_AMAE <- ranking_AMAE[
  order(ranking_AMAE$AMAE_medio),
]

rownames(ranking_AMAE) <- NULL

ranking_AMAE_reducida <- ranking_AMAE # para poder mostrar en informe

cat("\n========================================\n")
cat("RANKING AMAE - Base reducida\n")
cat("========================================\n")

ranking_AMAE_reducida

# el mejor modelo es "NB_ORDMAP"

# COMPARACIÓN UNILATERAL DEL MEJOR MODELO CONTRA EL RESTO
##############################################################################

mejor_modelo <- ranking_AMAE[[1,1]]

modelos_restantes <- setdiff(
  colnames(AMAE_comparacion),
  mejor_modelo
)

resultado_unilateral <- data.frame(
  modelo_referencia = mejor_modelo,
  modelo_comparado = modelos_restantes,
  media_referencia = NA,
  media_comparado = NA,
  p_shapiro = NA,
  normalidad = NA,
  test_aplicado = NA,
  p_valor = NA
)

for (j in 1:length(modelos_restantes)) {
  
  modelo_comparado <- modelos_restantes[j]
  
  x <- AMAE_comparacion[[mejor_modelo]]
  y <- AMAE_comparacion[[modelo_comparado]]
  
  diferencias <- x - y
  
  resultado_unilateral$media_referencia[j] <- mean(x)
  resultado_unilateral$media_comparado[j] <- mean(y)
  
  # Shapiro-Wilk sobre diferencias (para testear Normalidad)
  test_shapiro <- shapiro.test(diferencias)
  
  resultado_unilateral$p_shapiro[j] <- test_shapiro$p.value
  
  # Queremos contrastar:
  # H1: AMAE_NB_ORDMAP < AMAE_modelo_comparado
  
  # si Normalidad --> t-test
  if (test_shapiro$p.value >= 0.05) {
    
    resultado_unilateral$normalidad[j] <- "Normalidad"
    
    test_diferencias <- t.test(
      x,
      y,
      paired = TRUE,
      alternative = "less"
    )
    
    resultado_unilateral$test_aplicado[j] <- "t-test apareado unilateral"
    resultado_unilateral$p_valor[j] <- test_diferencias$p.value
  }
  
  # si NO Normalidad --> Wilcoxon
  
  if (test_shapiro$p.value < 0.05) {
    
    resultado_unilateral$normalidad[j] <- "No normalidad"
    
    test_diferencias <- wilcox.test(
      x,
      y,
      paired = TRUE,
      alternative = "less",
      exact = FALSE
    )
    
    resultado_unilateral$test_aplicado[j] <- "Wilcoxon apareado unilateral"
    resultado_unilateral$p_valor[j] <- test_diferencias$p.value
  }
}

# Corrección por comparaciones múltiples
resultado_unilateral$p_valor_ajustado_Holm <- p.adjust(
  resultado_unilateral$p_valor,
  method = "holm"
)

resultado_unilateral$significativo <- resultado_unilateral$p_valor_ajustado_Holm < 0.05

resultado_unilateral_reducida <- resultado_unilateral # para poder mostrar en informe porque se sobreescribe

cat("\n========================================\n")
cat("COMPARACIÓN - Base reducida\n")
cat("========================================\n")

resultado_unilateral_reducida


## -------------------------------------------------------------------------------------------------------------
#################### BASE REDUCIDA: seleccionamos el modelo Naive Bayes con criterio Ord-MAP ####################################

AMAE_reducida <- AMAE_comparacion[["NB_ORDMAP"]]


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# COMPARACIÓN FINAL: BASE COMPLETA VS BASE REDUCIDA
##############################################################################

alpha <- 0.05
cat("AMAEs medios: \nBase reducida:", mean(AMAE_reducida), "\nBase completa", mean(AMAE_completa))
dif_AMAE <- AMAE_completa - AMAE_reducida

# Test de normalidad sobre las diferencias
cat("\nTest de Shapiro-Wilk:\n")

  # H0: las diferencias dif_AMAE siguen una distribución normal
  # H1: las diferencias dif_AMAE no siguen una distribución normal

shapiro_res <- shapiro.test(dif_AMAE)
print(shapiro_res)

cat("Se rechaza Normalidad, por lo tanto utilizamos un test wilcoxon")

  # Test de diferencias apareadas
cat("\nRealizamos un test unilateral para comprobar si AMAE_completa es significativamente inferior:")
  # Queremos contrastar:
  # H0: AMAE_base_reducida >= AMAE_base_completa
  # H1: AMAE_base_reducida <  AMAE_base_completa
  
test_res <- wilcox.test(
  AMAE_reducida,
  AMAE_completa,
  paired = TRUE,
  alternative = "less"
)

print(test_res)

cat("No podemos descartar igualdad, por lo que nos quedamos con el modelo más simple: NB_ORDMAP con base reducida")

# no podemos rechazar igualdad, por loq ue nso quedamos con el modelo más simple

#################### DECISIÓN FINAL: seleccionamos el modelo Naive Bayes con criterio Ord-MAP aplicado a la base de datos reducida, por parsimonia ####################################


## -------------------------------------------------------------------------------------------------------------
# Diferencias
dif_AMAE <- AMAE_reducida - AMAE_completa

# Medias
mean(AMAE_reducida)
mean(AMAE_completa)
mean(dif_AMAE)

# Normalidad de las diferencias
shapiro.test(dif_AMAE)

# Test no paramétrico pareado
wilcox.test(AMAE_reducida, AMAE_completa,
            paired = TRUE,
            alternative = "less",
            exact = FALSE)


## -------------------------------------------------------------------------------------------------------------
############################################################################## 
# ENTRENAR MODELO FINAL
##############################################################################

# estructura 

xarxa_NB_final <- naive.bayes(
  datos_reducida,
  variable_objetivo
)

# parámetros

xarxa_NB_final_estimada <- bn.fit(
  xarxa_NB_final,
  datos_reducida,
  method = "bayes",
  iss = 1
)

# objeto gRain (para inferencia)

xarxa_NB_final_grain <- as.grain(
  xarxa_NB_final_estimada
)

# DAG

graphviz.plot(
  xarxa_NB_final,
  layout = "dot",
  main = "Red bayesiana NB - Base reducida BIC"
)


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# TABLAS DE PROBABILIDAD CONDICIONADA
##############################################################################
cat("TABLAS DE PROBABILIDAD CONDICIONADA:")


for (nodo in names(xarxa_NB_final_estimada)) {
  cat("\n========================================\n")
  cat(nodo, "\n")
  cat("========================================\n")
  
  print(round(xarxa_NB_final_estimada[[nodo]]$prob, 4))
}


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# PROBABILIDADES POSTERIORES
##############################################################################

# variables_predictoras ya definido

# creamos una funcion para calcularlas
estudiar_variable <- function(red_grain, datos, variable_objetivo, variable) {
  
  niveles <- levels(datos[[variable]])
  
  resultados <- data.frame()
  
  for (nivel in niveles) {
    
    red_ev <- setEvidence(
      red_grain,
      nodes = variable,
      states = nivel
    )
    
    probs <- querygrain(
      red_ev,
      nodes = variable_objetivo,
      type = "marginal"
    )[[variable_objetivo]]
    
    fila <- data.frame(
      variable = variable,
      nivel = nivel,
      muy_baja = probs["muy_baja"],
      baja = probs["baja"],
      media = probs["media"],
      alta = probs["alta"]
    )
    
    resultados <- rbind(resultados, fila)
  }
  
  rownames(resultados) <- NULL
  
  return(resultados)
}

cat("PROBABILIDADES POSTERIORES:")

# aplicamos la función a cada variable
##############################################################################
# 1. CHANNEL
##############################################################################

res_channel <- estudiar_variable(
  red_grain = xarxa_NB_final_grain,
  datos = datos_reducida,
  variable_objetivo = "popularidad",
  variable = "channel"
)

cat("\n========================================\n")
cat("Variable: channel\n")
cat("========================================\n")
print(res_channel)


##############################################################################
# 2. IS_WEEKEND
##############################################################################

res_is_weekend <- estudiar_variable(
  red_grain = xarxa_NB_final_grain,
  datos = datos_reducida,
  variable_objetivo = "popularidad",
  variable = "is_weekend"
)

cat("\n========================================\n")
cat("Variable: is_weekend\n")
cat("========================================\n")
print(res_is_weekend)


##############################################################################
# 3. N_TOKENS_CONTENT_CAT
##############################################################################

res_n_tokens_content <- estudiar_variable(
  red_grain = xarxa_NB_final_grain,
  datos = datos_reducida,
  variable_objetivo = "popularidad",
  variable = "n_tokens_content_cat"
)

cat("\n========================================\n")
cat("Variable: n_tokens_content_cat\n")
cat("========================================\n")
print(res_n_tokens_content)


##############################################################################
# 4. NUM_HREFS_CAT
##############################################################################

res_num_hrefs <- estudiar_variable(
  red_grain = xarxa_NB_final_grain,
  datos = datos_reducida,
  variable_objetivo = "popularidad",
  variable = "num_hrefs_cat"
)

cat("\n========================================\n")
cat("Variable: num_hrefs_cat\n")
cat("========================================\n")
print(res_num_hrefs)


##############################################################################
# 5. KW_AVG_AVG_CAT
##############################################################################

res_kw_avg_avg <- estudiar_variable(
  red_grain = xarxa_NB_final_grain,
  datos = datos_reducida,
  variable_objetivo = "popularidad",
  variable = "kw_avg_avg_cat"
)

cat("\n========================================\n")
cat("Variable: kw_avg_avg_cat\n")
cat("========================================\n")
print(res_kw_avg_avg)


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# CAPACIDAD DISCRIMINATIVA
##############################################################################

# Creamos una función para estudiar el cambio medio de probabilidades de popularidad para cada nivel de las variables predictoras

capacidad_discriminativa <- function(variable) {
  
  cpt <- xarxa_NB_final_estimada[[variable]]$prob
  
  diferencias <- apply(
    cpt,
    1,
    function(x) max(x) - min(x)
  )
  
  return(mean(diferencias))
}


disc_channel <- capacidad_discriminativa("channel")

disc_is_weekend <- capacidad_discriminativa("is_weekend")

disc_n_tokens_content <- capacidad_discriminativa("n_tokens_content_cat")

disc_num_hrefs <- capacidad_discriminativa("num_hrefs_cat")

disc_kw_avg_avg <- capacidad_discriminativa("kw_avg_avg_cat")

# Resumen 

discriminacion <- data.frame(
  variable = c(
    "channel",
    "is_weekend",
    "n_tokens_content_cat",
    "num_hrefs_cat",
    "kw_avg_avg_cat"
  ),
  discriminacion_media = c(
    disc_channel,
    disc_is_weekend,
    disc_n_tokens_content,
    disc_num_hrefs,
    disc_kw_avg_avg
  )
)

discriminacion <- discriminacion[
  order(-discriminacion$discriminacion_media),
]

discriminacion


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# PREDICCIONES DEL MODELO FINAL
##############################################################################

variable_objetivo <- "popularidad"

niveles <- c(
  "muy_baja",
  "baja",
  "media",
  "alta"
)

inputs_final <- setdiff(
  names(datos_reducida),
  variable_objetivo
)

inputs_final


##############################################################################
# FUNCIÓN AUXILIAR PARA PREDECIR CASOS
# Como hay tres bloques de predicción así no tenemos que repetir código cada
# vez
##############################################################################

predecir_casos <- function(casos, info = NULL) {

  # Nos quedamos solo con las variables predictoras
  casos_pred <- casos[
    inputs_final
  ]
  
  # Mismos niveles que en la base de entrenamiento
  for (v in inputs_final) {
    
    casos_pred[[v]] <- factor(
      casos_pred[[v]],
      levels = levels(datos_reducida[[v]])
    )
  }
  
  # Predicción probabilística
  pred <- predict(
    xarxa_NB_final_grain,
    response = variable_objetivo,
    newdata = casos_pred,
    type = "distribution"
  )
  
  matriu_probs <- as.matrix(
    pred$pred[[variable_objetivo]]
  )
  
  matriu_probs <- matriu_probs[
    ,
    niveles,
    drop = FALSE
  ]
  
  # Predicción OrdMAP
  pred_OrdMAP <- apply(
    matriu_probs,
    1,
    OrdMAP,
    levels = niveles
  )
  
  pred_OrdMAP <- factor(
    pred_OrdMAP,
    levels = niveles,
    ordered = TRUE
  )
  
  # Predicción MAP
  pred_MAP <- colnames(matriu_probs)[
    max.col(
      matriu_probs,
      ties.method = "first"
    )
  ]
  
  pred_MAP <- factor(
    pred_MAP,
    levels = niveles,
    ordered = TRUE
  )
  
  # Probabilidades
  probabilidades <- as.data.frame(
    round(matriu_probs, 4)
  )
  
  names(probabilidades) <- paste0(
    "P_",
    names(probabilidades)
  )
  
  # Tabla final
  if (is.null(info)) {
    
    tabla <- cbind(
      prediccion_OrdMAP = pred_OrdMAP,
      prediccion_MAP = pred_MAP,
      probabilidades,
      casos_pred
    )
    
  } else {
    
    tabla <- cbind(
      info,
      prediccion_OrdMAP = pred_OrdMAP,
      prediccion_MAP = pred_MAP,
      probabilidades,
      casos_pred
    )
  }
  
  return(tabla)
}


##############################################################################
# 1) CASOS NUEVOS QUE NO APARECEN EN LA BASE DE DATOS
# Són combinaciones de casos que no han aparecido en nuestra BD categorizada
# y sería interesante verlos por si, por ejemplo, nos estubieramos dejando
# algo muy interesante
##############################################################################

# Niveles posibles de cada variable predictora
niv <- lapply(
  datos_reducida[inputs_final],
  levels
)

# Todas las combinaciones posibles
comb_posibles <- expand.grid(
  niv,
  stringsAsFactors = FALSE
)

names(comb_posibles) <- inputs_final

# Combinaciones existentes en la base
comb_exist <- unique(
  datos_reducida[inputs_final]
)

comb_pos_str <- apply(
  comb_posibles,
  1,
  paste,
  collapse = "||"
)

comb_exist_str <- apply(
  comb_exist,
  1,
  paste,
  collapse = "||"
)

# Combinaciones que no aparecen en la base
faltantes <- comb_posibles[
  !(comb_pos_str %in% comb_exist_str),
  ,
  drop = FALSE
]


info_nuevos <- data.frame(
  caso = paste0(
    "Caso nuevo ",
    1:nrow(faltantes)
  )
)

tabla_casos_nuevos <- predecir_casos(
  casos = faltantes,
  info = info_nuevos
)

tabla_casos_nuevos


##############################################################################
# 2) CASOS CREADOS MANUALMENTE
# Puede que estén en la BD o puede que no, el objetivo es crear casos de ejemplo
##############################################################################

casos_manuales <- data.frame(
  caso = c(
    "Perfil potencialmente favorable",
    "Perfil claramente desfavorable",
    "Perfil mixto"
  ),
  
  channel = c(
    "socmed",
    "world",
    "business"
  ),
  
  is_weekend = c(
    "1",
    "0",
    "0"
  ),
  
  n_tokens_content_cat = c(
    "corto",
    "largo",
    "largo"
  ),
  
  num_hrefs_cat = c(
    "muchos",
    "pocos_ninguno",
    "muchos"
  ),
  
  kw_avg_avg_cat = c(
    "alta",
    "baja",
    "alta"
  ),
  
  stringsAsFactors = FALSE
)

info_manuales <- casos_manuales[
  "caso"
]

tabla_casos_manuales <- predecir_casos(
  casos = casos_manuales,
  info = info_manuales
)

tabla_casos_manuales


##############################################################################
# 3) CASOS DE INTERÉS DE LA BASE DE DATOS
# Casos reales de la base de datos, que ya han sido clasificados, pero que
# nos resultaba interesante ver
##############################################################################

ids_interes <- c(
  30663, # Correcto: alta -> alta
  36871, # Correcto: muy_baja -> muy_baja
  25412, # Infraestimado: alta -> muy_baja
  10455, # Sobreestimado fuerte: muy_baja -> alta
  7473   # Sobreestimado leve: muy_baja -> baja
)

casos_bd <- datos_reducida[
  ids_interes,
  ,
  drop = FALSE
]

info_bd <- data.frame(
  caso = c(
    "Correcto: alta -> alta",
    "Correcto: muy_baja -> muy_baja",
    "Infraestimado: alta -> muy_baja",
    "Sobreestimado fuerte: muy_baja -> alta",
    "Sobreestimado leve: muy_baja -> baja"
  ),
  
  observada_validacion = c(
    "alta",
    "muy_baja",
    "alta",
    "muy_baja",
    "muy_baja"
  ),
  
  predicha_validacion = c(
    "alta",
    "muy_baja",
    "muy_baja",
    "alta",
    "baja"
  )
)

tabla_casos_bd <- predecir_casos(
  casos = casos_bd,
  info = info_bd
)

tabla_casos_bd



## -------------------------------------------------------------------------------------------------------------
##############################################################################
# ESTUDIO IMPORTANCIA VARIABLES (O IMPORTANCIA DE ASOCIACIÓN CON POPULARIDAD)
# Hacemos distancias de chi-cuadrado y v de cramer para poder "cuantificar"
# la asociación con popularidad y conocer las variables que más importan
##############################################################################

variable_objetivo <- "popularidad"

niveles <- c(
  "muy_baja",
  "baja",
  "media",
  "alta"
)

datos_reducida$popularidad <- factor(
  datos_reducida$popularidad,
  levels = niveles,
  ordered = TRUE
)

variables_predictoras <- setdiff(
  names(datos_reducida),
  variable_objetivo
)

resultados_importancia <- data.frame(
  variable = character(),
  chi_cuadrado = numeric(),
  grados_libertad = numeric(),
  p_valor = numeric(),
  p_valor_ajustado = numeric(),
  v_cramer = numeric(),
  pct_alta_min = numeric(),
  pct_alta_max = numeric(),
  diferencia_pct_alta = numeric(),
  stringsAsFactors = FALSE
)

for (v in variables_predictoras) {
  
  # Tabla de contingencia entre la variable predictora y popularidad,
  # para medir distancias y métricas de asociación
  tabla <- table(
    datos_reducida[[v]],
    datos_reducida[[variable_objetivo]]
  )
  
  # Chi cuadrado
  test_chi <- suppressWarnings(
    chisq.test(tabla)
  )
  
  # V de Cramer
  n_total <- sum(tabla)
  
  min_dim <- min(
    nrow(tabla) - 1,
    ncol(tabla) - 1
  )
  
  v_cramer <- sqrt(
    as.numeric(test_chi$statistic) /
      (n_total * min_dim)
  )
  
  # Porcentajes por fila
  porcentajes_fila <- prop.table(
    tabla,
    margin = 1
  )
  
  # Diferencia entre la categoría con más y menos porcentaje de popularidad alta
  if ("alta" %in% colnames(porcentajes_fila)) {
    
    pct_alta <- 100 * porcentajes_fila[, "alta"]
    pct_alta_min <- min(
      pct_alta,
      na.rm = TRUE
    )
    pct_alta_max <- max(
      pct_alta,
      na.rm = TRUE
    )
    
    diferencia_pct_alta <- pct_alta_max - pct_alta_min
  } else {
    
    pct_alta_min <- NA
    pct_alta_max <- NA
    diferencia_pct_alta <- NA
  }
  
  resultados_importancia <- rbind(
    resultados_importancia,
    data.frame(
      variable = v,
      chi_cuadrado = as.numeric(test_chi$statistic),
      grados_libertad = as.numeric(test_chi$parameter),
      p_valor = test_chi$p.value,
      p_valor_ajustado = NA,
      v_cramer = v_cramer,
      pct_alta_min = pct_alta_min,
      pct_alta_max = pct_alta_max,
      diferencia_pct_alta = diferencia_pct_alta,
      stringsAsFactors = FALSE
    )
  )
}

# Corrección por comparaciones múltiples
resultados_importancia$p_valor_ajustado <- p.adjust(
  resultados_importancia$p_valor,
  method = "holm"
)

# Ordenamos
resultados_importancia <- resultados_importancia[
  order(
    -resultados_importancia$v_cramer
  ),
]

# Redondeamoe
resultados_importancia$v_cramer <- round(
  resultados_importancia$v_cramer,
  4
)

resultados_importancia$p_valor <- signif(
  resultados_importancia$p_valor,
  4
)

resultados_importancia$p_valor_ajustado <- signif(
  resultados_importancia$p_valor_ajustado,
  4
)

resultados_importancia$pct_alta_min <- round(
  resultados_importancia$pct_alta_min,
  2
)

resultados_importancia$pct_alta_max <- round(
  resultados_importancia$pct_alta_max,
  2
)

resultados_importancia$diferencia_pct_alta <- round(
  resultados_importancia$diferencia_pct_alta,
  2
)

resultados_importancia


## -------------------------------------------------------------------------------------------------------------
##############################################################################
# GRÁFICOS
# Gráficos auxiliares para las diapos de la presentación
# Se guardan en png en la carpeta donde esté el script, por pantalla salen feos
##############################################################################

##############################################################################
# 1) GRÁFICOS 2x3 Y 3x5
##############################################################################

variable_objetivo <- "popularidad"

niveles_popularidad <- c(
  "muy_baja",
  "baja",
  "media",
  "alta"
)

cols_popularidad <- c(
  "Muy baja" = "#F5EBDD",
  "Baja"     = "#D9DEE5",
  "Media"    = "#C2D3DF",
  "Alta"     = "#4D7FBF"
)

cols_base <- c(
  "#F5EBDD",
  "#D9DEE5",
  "#C2D3DF",
  "#8CAACF",
  "#4D7FBF",
  "#B8C7D9",
  "#E8D8C8"
)

map_channel <- c(
  "other"         = "Otros",
  "lifestyle"    = "Estilo vida",
  "entertainment" = "Entretenimiento",
  "business"     = "Negocios",
  "socmed"       = "Redes sociales",
  "tech"         = "Tecnología",
  "world"        = "Mundo"
)

preparar_datos_graficos <- function(datos) {
  
  datos <- as.data.frame(datos)
  
  datos$popularidad <- factor(
    datos$popularidad,
    levels = niveles_popularidad,
    labels = c("Muy baja", "Baja", "Media", "Alta"),
    ordered = TRUE
  )
  
  datos$is_weekend <- factor(
    as.character(datos$is_weekend),
    levels = c("0", "1"),
    labels = c("No", "Sí")
  )
  
  channel_original <- as.character(datos$channel)
  
  channel_recodificado <- map_channel[
    channel_original
  ]
  
  channel_recodificado[
    is.na(channel_recodificado)
  ] <- channel_original[
    is.na(channel_recodificado)
  ]
  
  datos$channel <- factor(
    channel_recodificado,
    levels = c(
      "Otros",
      "Estilo vida",
      "Entretenimiento",
      "Negocios",
      "Redes sociales",
      "Tecnología",
      "Mundo"
    )
  )
  
  for (v in names(datos)) {
    
    datos[[v]] <- factor(
      datos[[v]]
    )
    
    levels(datos[[v]]) <- gsub(
      pattern = "_",
      replacement = " ",
      x = levels(datos[[v]])
    )
  }
  
  return(datos)
}

tema_grafico <- function(size_title = 10, size_x = 7) {
  
  theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "none",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(
        color = "black",
        size = size_title,
        face = "bold",
        hjust = 0.5
      ),
      axis.text.x = element_text(
        color = "black",
        size = size_x,
        angle = 25,
        hjust = 1
      ),
      axis.text.y = element_text(color = "black", size = 7),
      axis.title = element_text(color = "black", size = 8),
      axis.line = element_line(color = "black")
    )
}

crear_grafico_cruzado <- function(datos, variable, nombres) {
  
  titulo <- nombres[[variable]]
  
  if (variable == variable_objetivo) {
    
    p <- ggplot(
      datos,
      aes(
        x = .data[[variable]],
        fill = .data[[variable]]
      )
    ) +
      geom_bar(color = "black", linewidth = 0.2) +
      scale_fill_manual(values = cols_popularidad, name = "Popularidad")
    
  } else {
    
    p <- ggplot(
      datos,
      aes(
        x = .data[[variable]],
        fill = .data[[variable_objetivo]]
      )
    ) +
      geom_bar(position = "dodge", color = "black", linewidth = 0.2) +
      scale_fill_manual(values = cols_popularidad, name = "Popularidad")
  }
  
  p +
    labs(
      title = titulo,
      x = "",
      y = "Frecuencia"
    ) +
    tema_grafico(size_title = 12, size_x = 9)
}

crear_grafico_simple <- function(datos, variable, nombres) {
  
  titulo <- paste(
    strwrap(nombres[[variable]], width = 24),
    collapse = "\n"
  )
  
  datos_plot <- data.frame(
    categoria = droplevels(datos[[variable]])
  )
  
  niveles_variable <- levels(datos_plot$categoria)
  
  if (variable == variable_objetivo) {
    
    colores_variable <- cols_popularidad[niveles_variable]
    
  } else {
    
    colores_variable <- setNames(
      rep(cols_base, length.out = length(niveles_variable)),
      niveles_variable
    )
  }
  
  ggplot(
    datos_plot,
    aes(
      x = categoria,
      fill = categoria
    )
  ) +
    geom_bar(color = "black", linewidth = 0.2) +
    scale_fill_manual(values = colores_variable) +
    labs(
      title = titulo,
      x = "",
      y = "Frecuencia"
    ) +
    tema_grafico(size_title = 10, size_x = 7)
}


# ---------------------------------------------------------------------------
# Gráfico 2x3: base reducida BIC según popularidad
# ---------------------------------------------------------------------------

datos_graficos_bic <- preparar_datos_graficos(datos_reducida)

nombres_bic <- c(
  "popularidad"          = "Popularidad",
  "channel"              = "Canal",
  "is_weekend"           = "Fin de semana",
  "kw_avg_avg_cat"       = "Palabra clave avg. avg.",
  "n_tokens_content_cat" = "Tokens contenido",
  "num_hrefs_cat"        = "Hiperlinks"
)

variables_bic <- names(nombres_bic)

plots_bic <- lapply(
  variables_bic,
  crear_grafico_cruzado,
  datos = datos_graficos_bic,
  nombres = nombres_bic
)

graficos_bic_2x3 <- wrap_plots(
  plots_bic,
  ncol = 3
) +
  plot_annotation(
    title = "Distribución de popularidad y variables según popularidad de la base reducida",
    theme = theme(
      plot.title = element_text(
        color = "black",
        size = 16,
        face = "bold",
        hjust = 0.5
      ),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

graficos_bic_2x3

ggsave(
  filename = "graficos_base_reducida_2x3.png",
  plot = graficos_bic_2x3,
  width = 12,
  height = 8,
  dpi = 300,
  bg = "white"
)


# ---------------------------------------------------------------------------
# Gráfico 3x5: base completa categorizada
# ---------------------------------------------------------------------------

datos_graficos_completa <- preparar_datos_graficos(datos_completa)

nombres_completa <- c(
  "popularidad"                      = "Popularidad",
  "channel"                          = "Canal",
  "is_weekend"                       = "Fin de semana",
  "n_tokens_title_cat"               = "Tokens título",
  "n_tokens_content_cat"             = "Tokens contenido",
  "num_hrefs_cat"                    = "Hiperlinks",
  "num_imgs_cat"                     = "Imágenes",
  "num_videos_cat"                   = "Vídeos",
  "kw_avg_avg_cat"                   = "Palabra clave avg. avg.",
  "global_subjectivity_cat"          = "Subjetividad global",
  "avg_positive_polarity_cat"        = "Polaridad positiva avg.",
  "avg_negative_polarity_cat"        = "Polaridad negativa avg.",
  "title_subjectivity_cat"           = "Subjetividad título",
  "title_sentiment_polarity_cat"     = "Polaridad título",
  "abs_title_sentiment_polarity_cat" = "Intensidad polaridad título"
)

variables_completa <- names(nombres_completa)

plots_completa <- lapply(
  variables_completa,
  crear_grafico_simple,
  datos = datos_graficos_completa,
  nombres = nombres_completa
)

graficos_completa_3x5 <- wrap_plots(
  plots_completa,
  ncol = 5
) +
  plot_annotation(
    title = "Distribución de las variables categorizadas de la base completa",
    theme = theme(
      plot.title = element_text(
        color = "black",
        size = 16,
        face = "bold",
        hjust = 0.5
      ),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

graficos_completa_3x5

ggsave(
  filename = "graficos_base_completa_3x5.png",
  plot = graficos_completa_3x5,
  width = 16,
  height = 9,
  dpi = 300,
  bg = "white"
)


##############################################################################
# 2) REPRESENTACIÓN DEL MODELO TAN AIC
##############################################################################

datos_tan_aic <- data_cat_mb_aic

datos_tan_aic$popularidad <- factor(
  datos_tan_aic$popularidad,
  levels = niveles_popularidad,
  ordered = TRUE
)

datos_tan_aic[] <- lapply(
  datos_tan_aic,
  factor
)

datos_tan_aic <- droplevels(datos_tan_aic)

xarxa_TAN_aic <- tree.bayes(
  x = datos_tan_aic,
  training = variable_objetivo
)

png(
  filename = "xarxa_TAN_AIC.png",
  width = 1400,
  height = 1000,
  res = 150,
  bg = "white"
)

graphviz.plot(
  xarxa_TAN_aic,
  layout = "dot",
  main = "Red bayesiana TAN - Base completa AIC"
)

dev.off()



##############################################################################
# 3) REPRESENTACIÓN DEL MODELO NAIVE BAYES BIC
##############################################################################

png(
  filename = "xarxa_NB_BIC.png",
  width = 1400,
  height = 1000,
  res = 150,
  bg = "white"
)

graphviz.plot(
  xarxa_NB_final,
  layout = "dot",
  main = "Red bayesiana Naive Bayes - Base reducida BIC"
)

dev.off()



## -------------------------------------------------------------------------------------------------------------
# Guardar objetos para poder mostrar en informe
save.image(file = "PAC1_grup1.RData")

