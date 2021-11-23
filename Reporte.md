# Multivariate Adaptive Regression Splines (MARS)

### Introducción

El modelo MARS es un algoritmo de aprendizaje supervisado que funciona para regresión. En términos simples, es una versión modificada de regresión lineal, pero el modelo es más complicado ya que permite hacer diferentes pendientes para diferentes partes de la variable a estimar, y automaticamente modela términos no lineales e interacciones entre variables. Fue introducido originalmente en [1] por Friedman en 1991, y existen varias implementaciones del algoritmo, generalmente bajo el nombre de "Earth" [CITA] [CITA].  

### Comportamiento del Modelo



### Descripción del Modelo

MARS es un modelo de regresión no paramétrico que a través de funciones bisagra (*hinge functions*). El modelo en sí es un spline multivariante lineal: 
$$
f(X) = \beta_0 + \sum_{m=1}^M \beta_m h_m(X),
$$
donde $h(X) = (X_j - t)_+ = \max\{0, X_j - t\}$. 

### Implementación



### Ventajas y Desventajas



### Referencias

* [1] Friedman, J. H. (1991). "Multivariate Adaptive Regression Splines". *The Annals of Statistics*. **19** (1): 1–67.





https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full

https://towardsdatascience.com/mars-multivariate-adaptive-regression-splines-how-to-improve-on-linear-regression-e1e7a63c5eae

https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline

https://rubenfcasal.github.io/aprendizaje_estadistico/mars.html

https://web.stanford.edu/~hastie/Papers/ESLII.pdf