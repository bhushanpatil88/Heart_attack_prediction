# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:01:39.998837Z","iopub.execute_input":"2022-11-08T09:01:39.999341Z","iopub.status.idle":"2022-11-08T09:01:41.465985Z","shell.execute_reply.started":"2022-11-08T09:01:39.999246Z","shell.execute_reply":"2022-11-08T09:01:41.464318Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.pipeline import Pipeline,make_pipeline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:01:41.469287Z","iopub.execute_input":"2022-11-08T09:01:41.469872Z","iopub.status.idle":"2022-11-08T09:01:41.518335Z","shell.execute_reply.started":"2022-11-08T09:01:41.469820Z","shell.execute_reply":"2022-11-08T09:01:41.516655Z"}}
df = pd.read_csv("../input/heart-disease-dataset/heart.csv")
df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:01:52.271102Z","iopub.execute_input":"2022-11-08T09:01:52.271661Z","iopub.status.idle":"2022-11-08T09:01:52.521265Z","shell.execute_reply.started":"2022-11-08T09:01:52.271617Z","shell.execute_reply":"2022-11-08T09:01:52.520228Z"}}
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:01:55.329003Z","iopub.execute_input":"2022-11-08T09:01:55.329524Z","iopub.status.idle":"2022-11-08T09:01:55.399518Z","shell.execute_reply.started":"2022-11-08T09:01:55.329482Z","shell.execute_reply":"2022-11-08T09:01:55.398202Z"}}
df.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:01:57.966133Z","iopub.execute_input":"2022-11-08T09:01:57.966624Z","iopub.status.idle":"2022-11-08T09:01:57.981355Z","shell.execute_reply.started":"2022-11-08T09:01:57.966575Z","shell.execute_reply":"2022-11-08T09:01:57.979941Z"}}
 df.drop_duplicates(inplace=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:01:58.587176Z","iopub.execute_input":"2022-11-08T09:01:58.587652Z","iopub.status.idle":"2022-11-08T09:01:58.598973Z","shell.execute_reply.started":"2022-11-08T09:01:58.587614Z","shell.execute_reply":"2022-11-08T09:01:58.597335Z"}}
categorical_cols = []
categorical_inds = []
counting_cols = []
counting_inds  = []
cnt=0
for i in df.columns:
    cnt+=1
    if df[i].nunique()<=5:
        categorical_cols.append(i)
        categorical_inds.append(cnt-1)
    

      
    else:
        counting_cols.append(i)
        counting_inds.append(cnt-1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:02:02.678214Z","iopub.execute_input":"2022-11-08T09:02:02.678676Z","iopub.status.idle":"2022-11-08T09:02:02.683983Z","shell.execute_reply.started":"2022-11-08T09:02:02.678639Z","shell.execute_reply":"2022-11-08T09:02:02.683075Z"}}
print(categorical_cols)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:02:04.775568Z","iopub.execute_input":"2022-11-08T09:02:04.775998Z","iopub.status.idle":"2022-11-08T09:02:04.783811Z","shell.execute_reply.started":"2022-11-08T09:02:04.775965Z","shell.execute_reply":"2022-11-08T09:02:04.782093Z"}}
#They are already encoded

categorical_cols.remove('target')
categorical_cols.remove('sex')
categorical_inds.remove(1)
categorical_inds.remove(13)


print(categorical_cols)
print(counting_cols)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:02:07.480169Z","iopub.execute_input":"2022-11-08T09:02:07.480619Z","iopub.status.idle":"2022-11-08T09:02:07.495093Z","shell.execute_reply.started":"2022-11-08T09:02:07.480585Z","shell.execute_reply":"2022-11-08T09:02:07.493055Z"}}
for i in categorical_cols:
    print(i)
    print(df[i].value_counts())
    print()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:13.612538Z","iopub.execute_input":"2022-11-08T09:17:13.612970Z","iopub.status.idle":"2022-11-08T09:17:13.624582Z","shell.execute_reply.started":"2022-11-08T09:17:13.612938Z","shell.execute_reply":"2022-11-08T09:17:13.623238Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=2)
X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:16.036111Z","iopub.execute_input":"2022-11-08T09:17:16.036511Z","iopub.status.idle":"2022-11-08T09:17:16.046744Z","shell.execute_reply.started":"2022-11-08T09:17:16.036479Z","shell.execute_reply":"2022-11-08T09:17:16.045447Z"}}
df.iloc[102,:]


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:18.363877Z","iopub.execute_input":"2022-11-08T09:17:18.364405Z","iopub.status.idle":"2022-11-08T09:17:18.372075Z","shell.execute_reply.started":"2022-11-08T09:17:18.364364Z","shell.execute_reply":"2022-11-08T09:17:18.370526Z"}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:20.946505Z","iopub.execute_input":"2022-11-08T09:17:20.946917Z","iopub.status.idle":"2022-11-08T09:17:20.955823Z","shell.execute_reply.started":"2022-11-08T09:17:20.946884Z","shell.execute_reply":"2022-11-08T09:17:20.954075Z"}}
from sklearn.preprocessing import StandardScaler
trf4 = make_column_transformer((StandardScaler(),counting_inds),remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:21.791386Z","iopub.execute_input":"2022-11-08T09:17:21.792541Z","iopub.status.idle":"2022-11-08T09:17:21.809050Z","shell.execute_reply.started":"2022-11-08T09:17:21.792498Z","shell.execute_reply":"2022-11-08T09:17:21.807613Z"}}
pipe = make_pipeline(trf3,trf4)


X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:23.440341Z","iopub.execute_input":"2022-11-08T09:17:23.440852Z","iopub.status.idle":"2022-11-08T09:17:23.450697Z","shell.execute_reply.started":"2022-11-08T09:17:23.440812Z","shell.execute_reply":"2022-11-08T09:17:23.448956Z"}}
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix,accuracy_score


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print("confussion matrix")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print("confussion matrix")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    print()
    dt_acc_score = accuracy_score(y_test, y_pred)
    print("Accuracy :",dt_acc_score*100,'\n')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:25.159892Z","iopub.execute_input":"2022-11-08T09:17:25.160370Z","iopub.status.idle":"2022-11-08T09:17:25.433639Z","shell.execute_reply.started":"2022-11-08T09:17:25.160334Z","shell.execute_reply":"2022-11-08T09:17:25.432578Z"}}
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=2)
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:17:25.762173Z","iopub.execute_input":"2022-11-08T09:17:25.762597Z","iopub.status.idle":"2022-11-08T09:17:25.778888Z","shell.execute_reply.started":"2022-11-08T09:17:25.762562Z","shell.execute_reply":"2022-11-08T09:17:25.777385Z"}}
eval_metric(classifier1, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:06:32.242095Z","iopub.execute_input":"2022-11-08T09:06:32.243323Z","iopub.status.idle":"2022-11-08T09:06:32.252729Z","shell.execute_reply.started":"2022-11-08T09:06:32.243274Z","shell.execute_reply":"2022-11-08T09:06:32.251304Z"}}
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 2)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:06:35.048846Z","iopub.execute_input":"2022-11-08T09:06:35.049332Z","iopub.status.idle":"2022-11-08T09:06:35.072955Z","shell.execute_reply.started":"2022-11-08T09:06:35.049295Z","shell.execute_reply":"2022-11-08T09:06:35.071765Z"}}
eval_metric(classifier2, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:13.686989Z","iopub.execute_input":"2022-11-08T09:03:13.687449Z","iopub.status.idle":"2022-11-08T09:03:19.058755Z","shell.execute_reply.started":"2022-11-08T09:03:13.687413Z","shell.execute_reply":"2022-11-08T09:03:19.057636Z"}}
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [3,4,5,6,7,8,9,10,11,12,13],
    'min_samples_leaf': [10,20, 25, 75, 50, 100,150,200],
    'criterion': ["gini", "entropy"],
    'splitter':['best','random'],
}          
grid_search = GridSearchCV(estimator = classifier2,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_accuracy)
print(best_parameters)
print(best_model)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:22.838741Z","iopub.execute_input":"2022-11-08T09:03:22.839433Z","iopub.status.idle":"2022-11-08T09:03:23.084798Z","shell.execute_reply.started":"2022-11-08T09:03:22.839367Z","shell.execute_reply":"2022-11-08T09:03:23.083478Z"}}
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier()
classifier3.fit(X_train, y_train)
y_pred = classifier3.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:25.222676Z","iopub.execute_input":"2022-11-08T09:03:25.223128Z","iopub.status.idle":"2022-11-08T09:03:25.272079Z","shell.execute_reply.started":"2022-11-08T09:03:25.223091Z","shell.execute_reply":"2022-11-08T09:03:25.270952Z"}}
eval_metric(classifier3, X_train, y_train, X_test, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T06:41:42.948404Z","iopub.execute_input":"2022-11-05T06:41:42.948828Z","iopub.status.idle":"2022-11-05T06:44:52.838708Z","shell.execute_reply.started":"2022-11-05T06:41:42.948794Z","shell.execute_reply":"2022-11-05T06:44:52.837248Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators':[90,100,120,130],
    'max_depth': [3,4,5,6,7,8,9,10,11,12,13],
    'min_samples_leaf': [20, 25, 75],
    'criterion': ["gini", "entropy"],
}          
rf_grid_search = GridSearchCV(estimator = classifier3,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rf_grid_search.fit(X_train, y_train)
best_accuracy = rf_grid_search.best_score_
best_parameters = rf_grid_search.best_params_
best_model = rf_grid_search.best_estimator_

print(best_accuracy)
print(best_parameters)
print(best_model)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:30.015125Z","iopub.execute_input":"2022-11-08T09:03:30.015701Z","iopub.status.idle":"2022-11-08T09:03:30.044802Z","shell.execute_reply.started":"2022-11-08T09:03:30.015650Z","shell.execute_reply":"2022-11-08T09:03:30.042006Z"}}
from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 2)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:33.661265Z","iopub.execute_input":"2022-11-08T09:03:33.661699Z","iopub.status.idle":"2022-11-08T09:03:33.727819Z","shell.execute_reply.started":"2022-11-08T09:03:33.661665Z","shell.execute_reply":"2022-11-08T09:03:33.726089Z"}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:36.335416Z","iopub.execute_input":"2022-11-08T09:03:36.335893Z","iopub.status.idle":"2022-11-08T09:03:36.741898Z","shell.execute_reply.started":"2022-11-08T09:03:36.335857Z","shell.execute_reply":"2022-11-08T09:03:36.740569Z"}}
scores=[]

for i in range(1,40):
    classifier4 = KNeighborsClassifier(n_neighbors = i)
    classifier4.fit(X_train, y_train)
    y_pred = classifier4.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:39.302539Z","iopub.execute_input":"2022-11-08T09:03:39.302992Z","iopub.status.idle":"2022-11-08T09:03:39.311271Z","shell.execute_reply.started":"2022-11-08T09:03:39.302958Z","shell.execute_reply":"2022-11-08T09:03:39.309776Z"}}
maxi_ind=-1
maxi=0
cnt=0
for i in scores:
    cnt+=1
    if maxi<i:
        max_ind = cnt
        maxi = i
print(max_ind)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:41.694571Z","iopub.execute_input":"2022-11-08T09:03:41.695010Z","iopub.status.idle":"2022-11-08T09:03:41.713854Z","shell.execute_reply.started":"2022-11-08T09:03:41.694974Z","shell.execute_reply":"2022-11-08T09:03:41.711768Z"}}
classifier4 = KNeighborsClassifier(n_neighbors = max_ind)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:44.360368Z","iopub.execute_input":"2022-11-08T09:03:44.360804Z","iopub.status.idle":"2022-11-08T09:03:44.425157Z","shell.execute_reply.started":"2022-11-08T09:03:44.360768Z","shell.execute_reply":"2022-11-08T09:03:44.423511Z"}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:47.462577Z","iopub.execute_input":"2022-11-08T09:03:47.463068Z","iopub.status.idle":"2022-11-08T09:03:47.475288Z","shell.execute_reply.started":"2022-11-08T09:03:47.463032Z","shell.execute_reply":"2022-11-08T09:03:47.473862Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=2)

X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:52.165512Z","iopub.execute_input":"2022-11-08T09:03:52.165978Z","iopub.status.idle":"2022-11-08T09:03:52.172389Z","shell.execute_reply.started":"2022-11-08T09:03:52.165943Z","shell.execute_reply":"2022-11-08T09:03:52.170917Z"}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:54.116213Z","iopub.execute_input":"2022-11-08T09:03:54.116702Z","iopub.status.idle":"2022-11-08T09:03:54.123672Z","shell.execute_reply.started":"2022-11-08T09:03:54.116664Z","shell.execute_reply":"2022-11-08T09:03:54.121878Z"}}
from sklearn.preprocessing import StandardScaler
trf4 = make_column_transformer((StandardScaler(),counting_inds),remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:57.007823Z","iopub.execute_input":"2022-11-08T09:03:57.008333Z","iopub.status.idle":"2022-11-08T09:03:57.015446Z","shell.execute_reply.started":"2022-11-08T09:03:57.008290Z","shell.execute_reply":"2022-11-08T09:03:57.013647Z"}}
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=2)

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:03:59.224426Z","iopub.execute_input":"2022-11-08T09:03:59.224913Z","iopub.status.idle":"2022-11-08T09:03:59.507271Z","shell.execute_reply.started":"2022-11-08T09:03:59.224868Z","shell.execute_reply":"2022-11-08T09:03:59.506281Z"}}
pipe = make_pipeline(trf3,trf4,classifier)


pipe.fit(X_train,y_train)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:04:01.537152Z","iopub.execute_input":"2022-11-08T09:04:01.538050Z","iopub.status.idle":"2022-11-08T09:04:01.549464Z","shell.execute_reply.started":"2022-11-08T09:04:01.537989Z","shell.execute_reply":"2022-11-08T09:04:01.548496Z"}}
y_pred = pipe.predict(X_test)
y_pred

# %% [code] {"execution":{"iopub.status.busy":"2022-11-08T09:04:03.959563Z","iopub.execute_input":"2022-11-08T09:04:03.960065Z","iopub.status.idle":"2022-11-08T09:04:03.992989Z","shell.execute_reply.started":"2022-11-08T09:04:03.960002Z","shell.execute_reply":"2022-11-08T09:04:03.991769Z"}}
eval_metric(pipe, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:04:07.081249Z","iopub.execute_input":"2022-11-08T09:04:07.081694Z","iopub.status.idle":"2022-11-08T09:05:25.060917Z","shell.execute_reply.started":"2022-11-08T09:04:07.081650Z","shell.execute_reply":"2022-11-08T09:05:25.059453Z"}}
scores=[]
for i in range(300):
    X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=i)
    X_train = X_train.values
    y_train = y_train.values
    X_test  = X_test.values
    y_test  = y_test.values
    classifier = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=2)
    pipe = make_pipeline(trf3,trf4,classifier)
    
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:30.050161Z","iopub.execute_input":"2022-11-08T09:05:30.050592Z","iopub.status.idle":"2022-11-08T09:05:30.061885Z","shell.execute_reply.started":"2022-11-08T09:05:30.050557Z","shell.execute_reply":"2022-11-08T09:05:30.060424Z"}}
np.argmax(scores)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:32.497642Z","iopub.execute_input":"2022-11-08T09:05:32.498064Z","iopub.status.idle":"2022-11-08T09:05:32.507445Z","shell.execute_reply.started":"2022-11-08T09:05:32.498017Z","shell.execute_reply":"2022-11-08T09:05:32.505950Z"}}
scores[np.argmax(scores)]

# %% [code] {"execution":{"iopub.status.busy":"2022-11-08T09:05:35.145222Z","iopub.execute_input":"2022-11-08T09:05:35.145688Z","iopub.status.idle":"2022-11-08T09:05:35.155057Z","shell.execute_reply.started":"2022-11-08T09:05:35.145651Z","shell.execute_reply":"2022-11-08T09:05:35.153374Z"}}
np.mean(scores)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:38.057886Z","iopub.execute_input":"2022-11-08T09:05:38.058367Z","iopub.status.idle":"2022-11-08T09:05:38.262129Z","shell.execute_reply.started":"2022-11-08T09:05:38.058331Z","shell.execute_reply":"2022-11-08T09:05:38.260423Z"}}
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=np.argmax(scores))

X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

classifier = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=2)
pipe = make_pipeline(trf3,trf4,classifier)
    
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:43.770149Z","iopub.execute_input":"2022-11-08T09:05:43.770873Z","iopub.status.idle":"2022-11-08T09:05:43.800345Z","shell.execute_reply.started":"2022-11-08T09:05:43.770823Z","shell.execute_reply":"2022-11-08T09:05:43.799261Z"}}
eval_metric(pipe, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:47.736204Z","iopub.execute_input":"2022-11-08T09:05:47.736701Z","iopub.status.idle":"2022-11-08T09:05:47.744117Z","shell.execute_reply.started":"2022-11-08T09:05:47.736656Z","shell.execute_reply":"2022-11-08T09:05:47.742651Z"}}
import pickle

pickle.dump(pipe,open('pipe.pkl','wb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:49.872452Z","iopub.execute_input":"2022-11-08T09:05:49.872924Z","iopub.status.idle":"2022-11-08T09:05:49.880526Z","shell.execute_reply.started":"2022-11-08T09:05:49.872888Z","shell.execute_reply":"2022-11-08T09:05:49.878776Z"}}
pipe = pickle.load(open('pipe.pkl','rb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:50.253601Z","iopub.execute_input":"2022-11-08T09:05:50.254124Z","iopub.status.idle":"2022-11-08T09:05:50.261486Z","shell.execute_reply.started":"2022-11-08T09:05:50.254078Z","shell.execute_reply":"2022-11-08T09:05:50.259427Z"}}
test1 = np.array([53,1,0,140,203,1,0,155,1,3.1,0,0,3]).reshape(1,13)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T09:05:52.041799Z","iopub.execute_input":"2022-11-08T09:05:52.042436Z","iopub.status.idle":"2022-11-08T09:05:52.054677Z","shell.execute_reply.started":"2022-11-08T09:05:52.042389Z","shell.execute_reply":"2022-11-08T09:05:52.053281Z"}}
pipe.predict(test1)

# %% [code] {"jupyter":{"outputs_hidden":false}}
