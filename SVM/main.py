import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
for dirname, _, filenames in os.walk("/folder_location"):
    for filename in filenames:
        #Loading Data
        print(os.path.join(dirname, filename))
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        train.head()
        test.head()
        train.info()
        #Plottring digit label distribution
        sns.countplot(train["label"])
        plt.plot(figure=(16, 10))
        g = sns.countplot(train["label"], palette='icefire')
        plt.title('NUmber of digit classes')
        train.label.astype('category').value_counts()
        plt.show()
        #Pre-processing and normalizing dataset
        source = train.drop(["label"], axis=1)
        target = train["label"]
        source = source / 255.0
        x_train, x_val, y_train, y_val = train_test_split(source, target, test_size=0.25, random_state=2021)
        print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
        x_train.head()
        y_train.head()
        #Model Creation
        mdl = SVC(C=600, kernel='rbf', random_state=21, gamma="scale", verbose=True)
        #Training the Model
        mdl.fit(x_train, y_train)
        #Predicting model with validation data
        predicted = mdl.predict(x_val)
        predicted
        #Measuring accuracy score and plotting confusion matrix
        cm = confusion_matrix(predicted, y_val)

        disp = plot_confusion_matrix(mdl, x_val, y_val,
                                     cmap=plt.cm.Blues)
        plt.show()
        print("accuracy", metrics.accuracy_score(y_val, predicted))
        #Testing the trained model and Loading test result
        test = test / 255.0
        print(f"THe test data is \n {test.shape}")
        y_pred = mdl.predict(test)
        submission = {}
        submission['ImageId'] = range(1, 28001)
        submission['Label'] = y_pred
        submission = pd.DataFrame(submission)

        submission = submission[['ImageId', 'Label']]
        submission = submission.sort_values(['ImageId'])
        submission.to_csv("submisision.csv", index=False)
        print(submission['Label'].value_counts().sort_index())
        #Saving the Trained Model
        filename = 'SVM_model.sav'
        pickle.dump(mdl, open(filename, 'wb'))







        break
    break

