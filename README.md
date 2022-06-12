# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset.
4. From sklearn.tree import DecisionTreeRegressor and fir the model. 5.Find the mean square error and r squared score value of the model.
5. Check the trained model. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Subanu. K
RegisterNumber:  212219040152
*/
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x= data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2= metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```


## Output:
Initial Dataset:

![img11](https://user-images.githubusercontent.com/87663343/173235475-406cbe7a-bd7e-4a6c-8b03-7e8095f770e6.png)

Dataset Information:

![img12](https://user-images.githubusercontent.com/87663343/173235508-3d255084-5355-4d70-860b-776c6fc51f82.png)

Null Dataset:

![img13](https://user-images.githubusercontent.com/87663343/173235539-1d4d7c6b-d354-4146-930e-bc2447df3344.png)

Encoded Dataset:

![img14](https://user-images.githubusercontent.com/87663343/173235567-0cf5f31b-7f1d-4f9d-ac58-87381cc8795d.png)

Mean Square Error Value:

![img15](https://user-images.githubusercontent.com/87663343/173235606-fe1d923d-1597-4cb6-927b-94796ae0e348.png)

R Squared Score:

![img 16](https://user-images.githubusercontent.com/87663343/173235635-919ea648-cc67-46cb-a2f5-b8713c12cae3.png)

Predicted Value:

![img17](https://user-images.githubusercontent.com/87663343/173235667-e630c246-85c3-4abd-9c27-72fa37936afa.png)







## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
