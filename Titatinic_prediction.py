__author__ = 'Jie'
"""
Titanic: Machine Learning from Disaster
Predict survival on the Titanic and get familiar with ML basics!
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.externals import joblib

## read data from .csv
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

class CleanData():

    def title_corr(self,title):
        ### function used for title cleaning.
        title=title.split(',')[1].split('.')[0].strip()
        if title =='Mr' or title =='Mrs' or title =='Miss':
            # noted that, do not use the type of title =='Mr' or 'Mrs'.
            newT=title
        elif title=='Dr' or title =='Rev' or title =='Col' or title =='Capt' or title =='Major':
            newT='Crew'
        elif title=='Jonkheer' or title =='the Countess' or title =='Sir' or title =='Master' or title =='Lady':
            newT='Noble'
        elif title=='Don':
            newT='Mr'
        elif title=='Mme' or title == 'Ms' or title=='Dona':
            newT='Mrs'
        elif title=='Mlle':
            newT='Miss'
        else:
            print ("title not inclued:", title)
        return newT

    def age_ave(self,df,pclass,sex,title):
        tmp=df.groupby(['Pclass','Sex','Title'])['Age'].median()
        return tmp[pclass][sex][title]
        # data_group.first()
        # data_group.describe()
        # data_group.count()
        # data_group.apply(print)

    def titleClean(self):
        ##  cleaning the column "Name", and create a new column "Title".
        ##  complex way utilize loop.
        titles_list=[]
        for name in train_data['Name']:
            tmp=name.split(',')[1].split('.')[0].strip()
            titles_list.append(tmp)
        for i,title in enumerate(titles_list):
            titles_list[i]=self.title_corr(title)
        m=len(titles_list)
        titles_set=set(titles_list)
        print (titles_set)
        train_data['Title']=titles_list

    def cleanData(self):
        # fill the missing values in "Fare", and combine the numbers to a "family" column.
        train_data["Family"]=train_data["SibSp"]+train_data["Parch"]
        test_data["Family"]=test_data["SibSp"]+test_data["Parch"]

        ## clean the fare data
        # fill the missing value in "Fare" with median value,
        # inplace=True: modify the original object!
        fare_median=train_data['Fare'].median() # the median value of this column
        fare_median1=test_data['Fare'].median() # the median value of this column
        train_data['Fare'].fillna(fare_median,inplace=True)
        test_data['Fare'].fillna(fare_median1,inplace=True)

        ### utlize the apply() function, easy way
        ###cleaning the column "Name", and create a new column "Title".
        train_data['Title']=train_data['Name'].apply(self.title_corr)
        test_data['Title']=test_data['Name'].apply(self.title_corr)

        for ind, row in train_data.iterrows():
            # iterrows() : will help you loop through each row of a dataframe.
            # It returns an iterator containing index of each
            #  row and the data in each row as a Series.
            # print(ind,row)
            if pd.isna(row['Age']):
                newAge=self.age_ave(train_data,row['Pclass'],row['Sex'],row['Title'])
                train_data.loc[ind,'Age']=newAge
        for ind, row in test_data.iterrows():
            if pd.isna(row['Age']):
                newAge=self.age_ave(test_data,row['Pclass'],row['Sex'],row['Title'])
                test_data.loc[ind,'Age']=newAge
        return (train_data, test_data)



class ModelFit():

    def modelFit_randomForest(self):
        cleanData=CleanData()
        X_train,X_test=cleanData.cleanData()
        y_train=train_data['Survived']

        # normalize the data
        scaler=MinMaxScaler()
        features=["Pclass", "Sex","Age","Family","Fare",'Title','Embarked']
        X_train=pd.get_dummies(train_data[features])
        X_test=pd.get_dummies(test_data[features])
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)

        # fit the model
        clf=SVC(C=10)
        clf.fit(X_train_scaled,y_train)
        # clf=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
        # clf.fit(X_train_scaled,y_train)
        predictions=clf.predict(X_test_scaled)

        # output the predicted values
        results=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})
        results.to_csv("results.csv",index=False)
        print ("success !")

        return clf

    def saveMode(self,clf):
        joblib.dump(clf,"D:/python-ml/kaggle_Titanic/titanic.pkl")
        print ("the trained model has been saved")

    def loadModel(self):
        clf1=joblib.load("D:/python-ml/kaggle_Titanic/titanic.pkl")
        print ("the trained model has been loaded")

def main():
    modelFit=ModelFit()
    clf=modelFit.modelFit_randomForest()
    modelFit.saveMode(clf)

    # some pandas notes
    ind=train_data.index
    cols=train_data.columns
    woman=train_data.loc[train_data.Sex=="female"]["Survived"]
    woman_ratio=sum(woman)/len(woman)
    man=train_data.loc[train_data.Sex=="male"]["Survived"]
    man_ratio=sum(man)/len(man)

if __name__ == '__main__':
    main()


