# MachineLearning_Python
import os

import pandas as pd

#Seaborn is a Python visualization library based on matplotlib.
import seaborn as sns
pd._version_


#returns current working directory
os.getcwd()

#Change working directory
os.chdir("C:\\Users\\Rk\\Downloads")

#Read file into the python

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()
titanic_train.describe()
titanic_train.groupby('Survived').size()
titanic_train.groupby(['Sex', 'Survived']).size()

#Type Casting

titanic_train['Survived'] = titanic_train['Survived'].astype('category')
titanic_train['Pclass'] = titanic_train['Pclass'].astype('category')
titanic_train['Sex'] = titanic_train['Sex'].astype('category')
tiatnic_train['Embarked'] = titanic_train['Embarked'].astype('category')

#Explore Univariate Contineous features visually
sns.boxplot(x='Fare', data = titanic_train)
sns.distplot(titanic_train['Fare'])
sns.distplot(titanic_train['Fare'], bins = 20, rug= True, kde=False)
sns.distplot(titanic_train['Fare'], bins = 100, kde = False)
sns.kdeplot(data = titanic_train['Fare'])
sns.kdeplot(data = titanic_train['Fare'], shade = True)

#Explore Univariate Categorical Features
titanic_train['Survived'].describe()
titanic_train['Survived'].value_counts()
pd.crosstab(index= titanic_train["Survived"], columns = "count")
pd.crosstab(index = titanic_train["Pclass"],columns = "count")
pd.crosstab(index = titanic_train["Sex"], columns= "count")

# Explore Univariate categorical features visually
sns.countplot(x='Survived', data = titanic_train)
sns.countplot(x = 'Pclass', data = titanic_train)


#Explore bivariate relationships: Categorical Vs Categorical
pd.crosstab(index= titanic_train['Survived'], columns = titanic_train['Sex'])
pd.crosstab(index = titanic_train['Survived'], columns = titanic_train['Pclass'], margins = True)
pd.crosstab(index = titanic_train['Survived'], columns = titanic_train['Embarked'], margins = True)

sns.factorplot(x="Survived", hue ="Sex", data=titanic_train, kind="count", size =6)
sns.factorplot(x="Pclass", hue ="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Ebarked", hue="Survived", data=titanic_train, kind="count", size=6)

#Explore bivariate Relationships: Categorical Vs Contineous
sns.factorplot(x="Fare", row = "Survived", data = titanic_train, kind="box", size=6)


#Read test file into the python

titanic_test = pd.read_csv('test.csv')


titanic_test.info()
titanic_test.shape
titanic_test.describe()

titanic_test['Survived'] = 0
titanic_test.Survived[titanic_test.sex == "female"] = 1
titanic_test.to_csv("Submission.csv", columns = ['PassengerId', 'Survived'], index = False)

