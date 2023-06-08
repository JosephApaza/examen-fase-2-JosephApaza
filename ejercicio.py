# Apellidos: Apaza Solis
# Nombres: Joseph Carlos
# Código: 2020240471

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

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

# Calcular el AUC-ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC:", auc_roc)

# Realizar validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5)
print("Resultados de validación cruzada:", cv_scores)
print("Precisión promedio de validación cruzada:", cv_scores.mean())

# Generar el informe de clasificación
classification_rep = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(classification_rep)

# Calcular la curva de precisión-recall
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

# Graficar la curva de precisión-recall
plt.plot(recall_curve, precision_curve)
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.title("Curva de Precisión-Recall")
plt.show()
