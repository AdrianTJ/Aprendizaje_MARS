# Multivariate Adaptive Regression Splines (MARS)

### Introducción

El modelo MARS es un algoritmo de aprendizaje supervisado que funciona para regresión. En términos simples, es una versión modificada de regresión lineal, pero el modelo es más complicado ya que permite hacer diferentes pendientes para diferentes partes de la variable a estimar, y automaticamente modela términos no lineales e interacciones entre variables. Fue introducido originalmente en [1] por Friedman en 1991, y existen varias implementaciones del algoritmo, generalmente bajo el nombre de "Earth" [CITA] [CITA].  

### Comportamiento del Modelo



### Descripción del Modelo

MARS es un modelo de regresión no paramétrico que a través de funciones bisagra (*hinge functions*). Se puede considerar como una modificación al método CART con el fin de mejorar su desempeño en el contexto de regresión. Es un modelo que es bueno para problemas de alta dimensionalidad. Para la siguiente sección, usamos la terminología y seguimos el desarrollo que se presenta en [1]  y en [2]. 

MARS se define con el uso de funciones que son lineales por partes, de la forma $(x - t)_+$ y $(t - x)_+$. Estas funciones toman el máximo entre 0 y el valor dentro de la funcion, por lo tanto, 
$$
(x - t)_+ = \max\{ 0, x - t \}= \begin{cases} 
x - t &\mbox{si } x > t, \\
0 & \mbox{en otro caso}. \end{cases}
$$
Cada función es lineal a trozos con un cambio de pendiente en el valor $t$, lo cual las hace splines lineales, y a cada par dividido en el valor $t$ se le llama un *par reflejado*. 

La idea del método es formar pares reflejados para cada variable de entrada $X_j$ con cambios de pendiente en el spline en cada valor observado $x_{ij}$. Por lo tanto, para cada variable de entrada o variable predictiva, se define un spline distinto. Por lo tanto, para una variable $X_j$, la colección de funciones base es 
$$
\mathcal{C} = \{ (X_j - t)_+, (t - X_j)_+ \}, \text{ con } t \in \{ x_{1,j}, x_{2,j}, \ldots x_{N,j} \}, \text{ y } \,\, j = 1,2, \ldots , p.
$$
Esto significa que si todos los valores de entrada son distintos, se tienen $2Np$ funciones base en total, y $2N$ divisiones en cada uno de los splines para cada variable. El modelo que utilizamos entonces para juntar todas las variables es uno aditivo, de forma parecida que se hace con regresión lineal: 
$$
f(X) = \beta_0 + \sum_{m=1}^M \beta_m h_m(X),
$$
donde cada función $h_m(X)$ es elemento de $\mathcal{C}$ o una combinación lineal de estas funciones base, pero para esta explicación inicial, usaremos solamente funciones que no involucran interacciones. Este modelo sin interacciones funciona de forma muy parecida a los árboles de decisión CART.

El proceso de construcción del modelo (denotado $\mathcal{M}$) empieza con el modelo base $\hat f(X) = \hat \beta_0 = h_0(X) = 1$, donde se define un término que funciona como intercepto al orígen, y se hace un procedimiento que se llama *forward*. Después de esto, todas las funciones en el conjunto $\mathcal{C}$ son candidatas de entrada a $\mathcal{M}$. Para cada observación $x_{ij}$, se genera un punto de corte descrito por un par reflejado: 
$$
h_1(X) = h(X_j - x_{ij}) \\
h_2(X) = h(x_{i,j} - X_j)
$$
y con esto, se ajusta el nuevo modelo involucrando estas funciones:
$$
\hat f(X) = \hat \beta_0 + \hat \beta_1 h_1(X) + \hat \beta_2 h_2(X).
$$
Como esto es una forma lineal en términos de cada una de las funciones $h_i$, se hace el ajuste de cada parámetro $\hat \beta_i$ minimizando la suma de cuadrados. 

En cada etapa de este procedimiento, se van agregando pares reflejados para cada una de las observaciones que se tienen, y eventualmente, se llega a un modelo final que incluye todos los posibles cortes definidos por cada observación. Claramente, esto llevará a sobreajuste de los datos, pero esto es lo que se está buscando en esta parte del procedimiento. Ya teniendo $\mathcal{M}$ con todas las interacciones posibles, se empieza la segunda parte del ajuste del modelo, que es la parte de podarlo. En este caso, se van eliminando los términos $h_i(X)$ iterativamente, empezando por el que produce el menor incremento en el error cuadrático residual cuando se quita. Este procedimiento produce un mejor modelo para cada tamaño $\lambda$, donde este modelo lo denotamos $\hat f_\lambda$. 

Hay varios procedimientos que se pueden utilizar para estimar el valor óptimo de $\lambda$, como validación cruzada o bootstrap, y probablemente el mejor de ellos sea análisis de *leave one out*, pero esto involucra un gran costo computacional. Para minimizar este problema de costo computacional, los modelos MARS generalmente utilizan un procedimiento de validación cruzada generalizada (GCV, por sus siglas en inglés). Este criterio se define como: 
$$
GCV(\lambda) = \frac{ \sum_{i=1}^N (y_i - \hat f_\lambda(x_i))^2 }{\left( \frac{1 - M(\lambda)}{N} \right)^2},
$$
donde el valor $M(\lambda)$ es el número de parámetros *efectivos* en el modelo, que depende del número de términos más el número de puntos de corte utilizados penalizado por un factor (2 en el caso aditivo que estamos explicando, 3 cuando hay interacciones). 

Ya que describimos el modelo base, podemos regresar y considerar cuales serían las diferencias al incorporar términos de interacción. Si se tiene el modelo base sin podar, podemos seguir iterando y agregando términos, pero esta vez, se considera como nueva función base todos los productos de una cierta función $h_m$ en el modelo $\mathcal{M}$ con uno de los pares reflejados que definimos en $\mathcal{C}$. Estos se agregan al modelo de la siguiente forma: 
$$
\hat f(X) = \hat \beta_0 + \sum_{m=1}^M \hat \beta_m h_m(X) + \hat \beta_{M+1} h_\ell(X) \cdot (X_j - t)_+ + \hat \beta_{M+2} h_\ell(X) \cdot (t-X_j)_+,
$$
para cada $h_\ell \in \mathcal{M}$, y donde $t = x_{ij}$ cualquiera. Otra manera de ver esto es que el modelo base es interactuar cada una de los pares reflejados para una variable $x_{ij}$ con el término incial del modelo, $h_0 = \hat \beta_0 = 1$. 

### Implementación



### Ventajas y Desventajas



### Referencias

* [1] Friedman, J. H. (1991). "Multivariate Adaptive Regression Splines". *The Annals of Statistics*. **19** (1): 1–67.





https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full

https://towardsdatascience.com/mars-multivariate-adaptive-regression-splines-how-to-improve-on-linear-regression-e1e7a63c5eae

https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline

[2] https://rubenfcasal.github.io/aprendizaje_estadistico/mars.html

https://web.stanford.edu/~hastie/Papers/ESLII.pdf