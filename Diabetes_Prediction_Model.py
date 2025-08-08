import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # EDA - Exploratory Data Analysis
    # Load the dataset using Pandas df-dataframe
    df = pd.read_csv('diabetes.csv')  
    Border = '-'*120
    print("Dimension : ", df.shape)

    print("First 5 entries : ")
    print(df.head())

    print("\nInformation about the Dataset : ")
    print(df.info())

    # print("\nChecking for the null values in the Dataset : ")
    # print(df.isna().sum())

    print("\nStatistical Data : ")
    print(df.describe())

    # Splitting the Dataset into Independent and Dependent Variables
    X = df.drop(columns='Outcome')
    Y = df["Outcome"]

    # Plotting the Distribution table of the 'Outcome' column
    # sns.histplot(data=df, x=Y, bins=5)
    # plt.xlabel("Distribution of the Dependent Variable")
    # plt.ylabel("Frequency of Distribution")
    # plt.grid()
    # plt.show()

    # Feature Importance
    # sns.pairplot(data=df, hue="Outcome")
    # plt.show()


    # Data Preprocessing
    # handling 0 values in the dataset with mean of that column
    imputer = SimpleImputer(missing_values=0, strategy = 'mean')
    df['Glucose'] = imputer.fit_transform(df[['Glucose']])
    df['BloodPressure'] = imputer.fit_transform(df[['BloodPressure']])
    df['SkinThickness'] = imputer.fit_transform(df[['SkinThickness']])
    df['Insulin'] = imputer.fit_transform(df[['Insulin']])
    df['BMI'] = imputer.fit_transform(df[['BMI']])

    # print("\nUpdated Dataset : ")
    # print(df.shape)
    # print(df.head())
    
    min_max_scaled = MinMaxScaler()
    x_scaled = min_max_scaled.fit_transform(X)
    # print("\n\nAfter Min Max Scaling : ")
    # print(x_scaled)


    # Model Building
    # train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(x_scaled, Y, test_size= 0.2, random_state=42)

    # KNN
    print(Border)
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy using KNN Algorithm : ", accuracy*100)

    clf = classification_report(Y_test, Y_pred)
    print("Classification Report : ")
    print(clf)

    Conf_matrix = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix : ")
    print(Conf_matrix)

    # DecisionTree
    print(Border)
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy using DecisionTree Algorithm : ", accuracy*100)   

    clf = classification_report(Y_test, Y_pred)
    print("Classification Report : ")
    print(clf) 

    Conf_matrix = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix : ")
    print(Conf_matrix)


    # Logistic Regression
    print(Border)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy using Logistic Regression Algorithm : ", accuracy*100)
    clf = classification_report(Y_test, Y_pred)
    print("Classification Report : ")
    print(clf)

    Conf_matrix = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix : ")
    print(Conf_matrix)
    

    # for k in range(2,16):
    #     model = KNeighborsClassifier(n_neighbors=k)
    #     model.fit(X_train, Y_train)
    #     Y_pred = model.predict(X_test)
        
    #     accuracy = accuracy_score(Y_test, Y_pred)
    #     print(f"Accuracy using KNN Algorithm for K = {k} : ", accuracy*100)


    print(Border)
    print("-"*50, "Program Terminated", "-"*50)
    print(Border)


if __name__ == "__main__":
    main()