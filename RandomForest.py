from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
import numpy as np

# Import data
iris_data = datasets.load_iris()

# Create Data Frame 
df = pd.DataFrame({"sepal length": iris_data.data[:,0],"sepal width": iris_data.data[:,1],"petal length": iris_data.data[:,2],"petal width": iris_data.data[:,3], "species": iris_data.target})

# Separeate our dependent and independent variables
X= df[["sepal length","sepal width","petal length","petal width"]]
Y= df[["species"]]

# Separate our dataframe into 80% for training and 20% for test
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.2) 

# Create model
forest_model = RandomForestClassifier()

# Train model
forest_model.fit(x_train, y_train)

# Create a prediction
prediction = forest_model.predict(x_test)

# Print results
y_test_values= y_test.values

for i in range(len(prediction)):
  print("y_test:",y_test_values[i][0],"- Prediction:", prediction[i])

mse_train=mean_squared_error(prediction,y_test)
mae_train=mean_absolute_error(prediction,y_test)

print("\nAccuracy -> ", metrics.accuracy_score(y_test,prediction))
print("MSE -> ", mse_train)
print("MAE -> ", mae_train)

variance = np.var(prediction)
bias = np.mean((np.mean(prediction)-y_test_values)** 2) - variance

print("Variance -> ", variance)
print("Bias -> ", bias)

print("\nEl modelo presenta un bias muy pequeño y a su vez una varianza alta, esto puede significar que el modelo esta overfitting, es decir que si se realiza algún cambio en el dataset, aunque sean pequeños, estos tendran grandes repercusiones. Por lo tanto intentaremos realizar un mejor balance entre esots realizando ajustes al modelo.")


# Separate our dataframe into 70% for training and 30% for test
x_train2,x_test2,y_train2,y_test2=train_test_split(X,Y, test_size=0.3) 

forest_model2 = RandomForestClassifier(random_state = 42, n_estimators=50, max_depth=10, max_features=3, criterion = 'entropy')
forest_model2.fit(x_train2, y_train2)
prediction2 = forest_model2.predict(x_test2)
predictionTrain2 = forest_model2.predict(x_train2)

y_test_values2= y_test2.values

for i in range(len(prediction2)):
  print("y_test:",y_test_values2[i][0],"- Prediction:", prediction2[i])

mse_train2=mean_squared_error(prediction2,y_test2)
mae_train2=mean_absolute_error(prediction2,y_test2)
variance2 = np.var(prediction2)
bias2 = np.mean((np.mean(prediction2)-y_test_values2)** 2) - variance

print("\nAccuracy train -> ", metrics.accuracy_score(y_train2,predictionTrain2))
print("Accuracy test -> ", metrics.accuracy_score(y_test2,prediction2))
print("MSE -> ", mse_train2)
print("Variance -> ", variance2)
print("Bias -> ", bias2)

print("\nCon los pequeños cambios realizados en los parámetros del modelo se logro que el bias aumentara un poquito más sin embargo esto puede cambiar debido a la aleatoridad de la seleción de los datos para entrenar y probar. Con estos cambios se sigue teniendo un accuracy alto del .95")
