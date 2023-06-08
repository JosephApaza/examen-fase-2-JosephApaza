# Apellidos: Apaza Solis
# Nombres: Joseph Carlos
# Código: 2020240471

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar los datos desde el archivo CSV con el separador correcto
data = pd.read_csv("divorcios.csv", delimiter=";")

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion)

# Calcular la precisión
precision = precision_score(y_test, y_pred)
print("Precisión:", precision)

# Calcular el recall (sensibilidad)
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calcular el F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

