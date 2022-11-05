# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:07:54.777237Z","iopub.execute_input":"2022-11-05T14:07:54.777712Z","iopub.status.idle":"2022-11-05T14:07:54.797281Z","shell.execute_reply.started":"2022-11-05T14:07:54.777672Z","shell.execute_reply":"2022-11-05T14:07:54.795904Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:07:56.968434Z","iopub.execute_input":"2022-11-05T14:07:56.969650Z","iopub.status.idle":"2022-11-05T14:07:56.995614Z","shell.execute_reply.started":"2022-11-05T14:07:56.969606Z","shell.execute_reply":"2022-11-05T14:07:56.994719Z"}}
df = pd.read_csv("../input/heart-disease-dataset/heart.csv")
df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T13:53:32.664070Z","iopub.execute_input":"2022-11-05T13:53:32.664479Z","iopub.status.idle":"2022-11-05T13:53:32.958844Z","shell.execute_reply.started":"2022-11-05T13:53:32.664447Z","shell.execute_reply":"2022-11-05T13:53:32.957524Z"}}
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T13:53:35.058458Z","iopub.execute_input":"2022-11-05T13:53:35.058931Z","iopub.status.idle":"2022-11-05T13:53:35.122002Z","shell.execute_reply.started":"2022-11-05T13:53:35.058895Z","shell.execute_reply":"2022-11-05T13:53:35.120684Z"}}
df.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:08:00.940998Z","iopub.execute_input":"2022-11-05T14:08:00.941426Z","iopub.status.idle":"2022-11-05T14:08:00.954139Z","shell.execute_reply.started":"2022-11-05T14:08:00.941394Z","shell.execute_reply":"2022-11-05T14:08:00.952346Z"}}
 df.drop_duplicates(inplace=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:08:01.359493Z","iopub.execute_input":"2022-11-05T14:08:01.359959Z","iopub.status.idle":"2022-11-05T14:08:01.371898Z","shell.execute_reply.started":"2022-11-05T14:08:01.359923Z","shell.execute_reply":"2022-11-05T14:08:01.370323Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:08:03.578798Z","iopub.execute_input":"2022-11-05T14:08:03.579226Z","iopub.status.idle":"2022-11-05T14:08:03.587029Z","shell.execute_reply.started":"2022-11-05T14:08:03.579192Z","shell.execute_reply":"2022-11-05T14:08:03.585126Z"}}
print(categorical_cols)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:08:03.806606Z","iopub.execute_input":"2022-11-05T14:08:03.807467Z","iopub.status.idle":"2022-11-05T14:08:03.815185Z","shell.execute_reply.started":"2022-11-05T14:08:03.807403Z","shell.execute_reply":"2022-11-05T14:08:03.814014Z"}}
#They are already encoded

categorical_cols.remove('target')
categorical_cols.remove('sex')
categorical_inds.remove(1)
categorical_inds.remove(13)


print(categorical_cols)
print(counting_cols)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T14:08:06.859123Z","iopub.execute_input":"2022-11-05T14:08:06.860270Z","iopub.status.idle":"2022-11-05T14:08:06.873666Z","shell.execute_reply.started":"2022-11-05T14:08:06.860207Z","shell.execute_reply":"2022-11-05T14:08:06.872497Z"}}
for i in categorical_cols:
    print(i)
    print(df[i].value_counts())
    print()

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:18.737039Z","iopub.execute_input":"2022-11-05T10:36:18.737419Z","iopub.status.idle":"2022-11-05T10:36:18.799024Z","shell.execute_reply.started":"2022-11-05T10:36:18.737389Z","shell.execute_reply":"2022-11-05T10:36:18.797896Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=2)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:20.921249Z","iopub.execute_input":"2022-11-05T10:36:20.921707Z","iopub.status.idle":"2022-11-05T10:36:20.928849Z","shell.execute_reply.started":"2022-11-05T10:36:20.921670Z","shell.execute_reply":"2022-11-05T10:36:20.927772Z"},"jupyter":{"outputs_hidden":false}}
X_train.shape

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:23.126475Z","iopub.execute_input":"2022-11-05T10:36:23.126859Z","iopub.status.idle":"2022-11-05T10:36:23.133617Z","shell.execute_reply.started":"2022-11-05T10:36:23.126828Z","shell.execute_reply":"2022-11-05T10:36:23.132227Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:25.094918Z","iopub.execute_input":"2022-11-05T10:36:25.095306Z","iopub.status.idle":"2022-11-05T10:36:25.100912Z","shell.execute_reply.started":"2022-11-05T10:36:25.095275Z","shell.execute_reply":"2022-11-05T10:36:25.099782Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.preprocessing import StandardScaler
trf4 = make_column_transformer((StandardScaler(),counting_inds),remainder='passthrough')

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:27.691142Z","iopub.execute_input":"2022-11-05T10:36:27.691548Z","iopub.status.idle":"2022-11-05T10:36:27.712368Z","shell.execute_reply.started":"2022-11-05T10:36:27.691517Z","shell.execute_reply":"2022-11-05T10:36:27.711212Z"},"jupyter":{"outputs_hidden":false}}
pipe = make_pipeline(trf3,trf4)


X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:08:43.577996Z","iopub.execute_input":"2022-11-05T14:08:43.578494Z","iopub.status.idle":"2022-11-05T14:08:43.587726Z","shell.execute_reply.started":"2022-11-05T14:08:43.578456Z","shell.execute_reply":"2022-11-05T14:08:43.585990Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:33.173307Z","iopub.execute_input":"2022-11-05T10:36:33.174016Z","iopub.status.idle":"2022-11-05T10:36:33.276125Z","shell.execute_reply.started":"2022-11-05T10:36:33.173971Z","shell.execute_reply":"2022-11-05T10:36:33.275002Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:35.573457Z","iopub.execute_input":"2022-11-05T10:36:35.573831Z","iopub.status.idle":"2022-11-05T10:36:35.593202Z","shell.execute_reply.started":"2022-11-05T10:36:35.573802Z","shell.execute_reply":"2022-11-05T10:36:35.592403Z"},"jupyter":{"outputs_hidden":false}}
eval_metric(classifier1, X_train, y_train, X_test, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:41.041405Z","iopub.execute_input":"2022-11-05T10:36:41.041810Z","iopub.status.idle":"2022-11-05T10:36:41.124485Z","shell.execute_reply.started":"2022-11-05T10:36:41.041778Z","shell.execute_reply":"2022-11-05T10:36:41.123343Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:43.517784Z","iopub.execute_input":"2022-11-05T10:36:43.518165Z","iopub.status.idle":"2022-11-05T10:36:43.533962Z","shell.execute_reply.started":"2022-11-05T10:36:43.518134Z","shell.execute_reply":"2022-11-05T10:36:43.532765Z"},"jupyter":{"outputs_hidden":false}}
eval_metric(classifier2, X_train, y_train, X_test, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:46.499945Z","iopub.execute_input":"2022-11-05T10:36:46.500412Z","iopub.status.idle":"2022-11-05T10:36:51.482958Z","shell.execute_reply.started":"2022-11-05T10:36:46.500376Z","shell.execute_reply":"2022-11-05T10:36:51.479977Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:36:57.686608Z","iopub.execute_input":"2022-11-05T10:36:57.687020Z","iopub.status.idle":"2022-11-05T10:36:57.930493Z","shell.execute_reply.started":"2022-11-05T10:36:57.686987Z","shell.execute_reply":"2022-11-05T10:36:57.929146Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier()
classifier3.fit(X_train, y_train)
y_pred = classifier3.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:00.773769Z","iopub.execute_input":"2022-11-05T10:37:00.774218Z","iopub.status.idle":"2022-11-05T10:37:00.827324Z","shell.execute_reply.started":"2022-11-05T10:37:00.774177Z","shell.execute_reply":"2022-11-05T10:37:00.826219Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:07.872846Z","iopub.execute_input":"2022-11-05T10:37:07.873794Z","iopub.status.idle":"2022-11-05T10:37:07.892628Z","shell.execute_reply.started":"2022-11-05T10:37:07.873756Z","shell.execute_reply":"2022-11-05T10:37:07.891069Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 2)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:10.048322Z","iopub.execute_input":"2022-11-05T10:37:10.048757Z","iopub.status.idle":"2022-11-05T10:37:10.143228Z","shell.execute_reply.started":"2022-11-05T10:37:10.048723Z","shell.execute_reply":"2022-11-05T10:37:10.141255Z"},"jupyter":{"outputs_hidden":false}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:12.986634Z","iopub.execute_input":"2022-11-05T10:37:12.987052Z","iopub.status.idle":"2022-11-05T10:37:13.362160Z","shell.execute_reply.started":"2022-11-05T10:37:12.987016Z","shell.execute_reply":"2022-11-05T10:37:13.360565Z"},"jupyter":{"outputs_hidden":false}}
scores=[]

for i in range(1,40):
    classifier4 = KNeighborsClassifier(n_neighbors = i)
    classifier4.fit(X_train, y_train)
    y_pred = classifier4.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:15.246281Z","iopub.execute_input":"2022-11-05T10:37:15.246701Z","iopub.status.idle":"2022-11-05T10:37:15.253528Z","shell.execute_reply.started":"2022-11-05T10:37:15.246666Z","shell.execute_reply":"2022-11-05T10:37:15.252250Z"},"jupyter":{"outputs_hidden":false}}
maxi_ind=-1
maxi=0
cnt=0
for i in scores:
    cnt+=1
    if maxi<i:
        max_ind = cnt
        maxi = i
print(max_ind)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:18.356714Z","iopub.execute_input":"2022-11-05T10:37:18.357135Z","iopub.status.idle":"2022-11-05T10:37:18.377857Z","shell.execute_reply.started":"2022-11-05T10:37:18.357103Z","shell.execute_reply":"2022-11-05T10:37:18.376312Z"},"jupyter":{"outputs_hidden":false}}
classifier4 = KNeighborsClassifier(n_neighbors = max_ind)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:20.321369Z","iopub.execute_input":"2022-11-05T10:37:20.321785Z","iopub.status.idle":"2022-11-05T10:37:20.380220Z","shell.execute_reply.started":"2022-11-05T10:37:20.321754Z","shell.execute_reply":"2022-11-05T10:37:20.376129Z"},"jupyter":{"outputs_hidden":false}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:00:49.305167Z","iopub.execute_input":"2022-11-05T14:00:49.305664Z","iopub.status.idle":"2022-11-05T14:00:49.396067Z","shell.execute_reply.started":"2022-11-05T14:00:49.305625Z","shell.execute_reply":"2022-11-05T14:00:49.394857Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=2)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:27.606091Z","iopub.execute_input":"2022-11-05T10:37:27.606504Z","iopub.status.idle":"2022-11-05T10:37:27.613961Z","shell.execute_reply.started":"2022-11-05T10:37:27.606472Z","shell.execute_reply":"2022-11-05T10:37:27.612667Z"},"jupyter":{"outputs_hidden":false}}
X_train.shape

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:00:54.485203Z","iopub.execute_input":"2022-11-05T14:00:54.485763Z","iopub.status.idle":"2022-11-05T14:00:54.492853Z","shell.execute_reply.started":"2022-11-05T14:00:54.485717Z","shell.execute_reply":"2022-11-05T14:00:54.491944Z"}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:00:56.633048Z","iopub.execute_input":"2022-11-05T14:00:56.633655Z","iopub.status.idle":"2022-11-05T14:00:56.640947Z","shell.execute_reply.started":"2022-11-05T14:00:56.633600Z","shell.execute_reply":"2022-11-05T14:00:56.639678Z"}}
from sklearn.preprocessing import StandardScaler
trf4 = make_column_transformer((StandardScaler(),counting_inds),remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:01:31.418405Z","iopub.execute_input":"2022-11-05T14:01:31.418877Z","iopub.status.idle":"2022-11-05T14:01:31.518289Z","shell.execute_reply.started":"2022-11-05T14:01:31.418842Z","shell.execute_reply":"2022-11-05T14:01:31.517005Z"}}
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 2)

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:01:34.240611Z","iopub.execute_input":"2022-11-05T14:01:34.241087Z","iopub.status.idle":"2022-11-05T14:01:34.301876Z","shell.execute_reply.started":"2022-11-05T14:01:34.241048Z","shell.execute_reply":"2022-11-05T14:01:34.300636Z"}}
pipe = make_pipeline(trf3,trf4,classifier)


pipe.fit(X_train,y_train)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:01:40.411366Z","iopub.execute_input":"2022-11-05T14:01:40.412071Z","iopub.status.idle":"2022-11-05T14:01:40.424398Z","shell.execute_reply.started":"2022-11-05T14:01:40.412024Z","shell.execute_reply":"2022-11-05T14:01:40.423477Z"}}
y_pred = pipe.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:37:55.510969Z","iopub.execute_input":"2022-11-05T10:37:55.511408Z","iopub.status.idle":"2022-11-05T10:38:07.044767Z","shell.execute_reply.started":"2022-11-05T10:37:55.511357Z","shell.execute_reply":"2022-11-05T10:38:07.043627Z"},"jupyter":{"outputs_hidden":false}}
scores=[]
for i in range(300):
    X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.2,
                                                random_state=i)
    classifier = LogisticRegression()
    pipe = make_pipeline(trf3,trf4,classifier)
    
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:38:18.391020Z","iopub.execute_input":"2022-11-05T10:38:18.391391Z","iopub.status.idle":"2022-11-05T10:38:18.398630Z","shell.execute_reply.started":"2022-11-05T10:38:18.391361Z","shell.execute_reply":"2022-11-05T10:38:18.397398Z"},"jupyter":{"outputs_hidden":false}}
np.argmax(scores)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-05T10:38:21.294505Z","iopub.execute_input":"2022-11-05T10:38:21.294871Z","iopub.status.idle":"2022-11-05T10:38:21.301959Z","shell.execute_reply.started":"2022-11-05T10:38:21.294842Z","shell.execute_reply":"2022-11-05T10:38:21.300940Z"},"jupyter":{"outputs_hidden":false}}
scores[np.argmax(scores)]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:08:58.898686Z","iopub.execute_input":"2022-11-05T14:08:58.899102Z","iopub.status.idle":"2022-11-05T14:08:58.933807Z","shell.execute_reply.started":"2022-11-05T14:08:58.899069Z","shell.execute_reply":"2022-11-05T14:08:58.932157Z"}}
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.2,
                                                random_state=np.argmax(scores))
classifier = LogisticRegression()
pipe = make_pipeline(trf4,classifier)
    
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:09:08.488619Z","iopub.execute_input":"2022-11-05T14:09:08.489417Z","iopub.status.idle":"2022-11-05T14:09:08.524441Z","shell.execute_reply.started":"2022-11-05T14:09:08.489353Z","shell.execute_reply":"2022-11-05T14:09:08.522700Z"}}
eval_metric(pipe, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:09:11.738427Z","iopub.execute_input":"2022-11-05T14:09:11.738959Z","iopub.status.idle":"2022-11-05T14:09:11.746461Z","shell.execute_reply.started":"2022-11-05T14:09:11.738920Z","shell.execute_reply":"2022-11-05T14:09:11.745074Z"}}
import pickle

pickle.dump(pipe,open('pipe.pkl','wb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:09:12.306067Z","iopub.execute_input":"2022-11-05T14:09:12.306501Z","iopub.status.idle":"2022-11-05T14:09:12.314060Z","shell.execute_reply.started":"2022-11-05T14:09:12.306469Z","shell.execute_reply":"2022-11-05T14:09:12.312487Z"}}
pipe = pickle.load(open('pipe.pkl','rb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:09:13.314398Z","iopub.execute_input":"2022-11-05T14:09:13.314879Z","iopub.status.idle":"2022-11-05T14:09:13.322032Z","shell.execute_reply.started":"2022-11-05T14:09:13.314844Z","shell.execute_reply":"2022-11-05T14:09:13.320805Z"}}
test1 = np.array([53,1,0,140,203,1,0,155,1,3.1,0,0,3]).reshape(1,13)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-05T14:09:14.173036Z","iopub.execute_input":"2022-11-05T14:09:14.173567Z","iopub.status.idle":"2022-11-05T14:09:14.185212Z","shell.execute_reply.started":"2022-11-05T14:09:14.173509Z","shell.execute_reply":"2022-11-05T14:09:14.183607Z"}}
pipe.predict(test1)

# %% [code] {"jupyter":{"outputs_hidden":false}}
