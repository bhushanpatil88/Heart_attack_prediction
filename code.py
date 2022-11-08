# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:31:42.839636Z","iopub.execute_input":"2022-11-08T16:31:42.840131Z","iopub.status.idle":"2022-11-08T16:31:42.847405Z","shell.execute_reply.started":"2022-11-08T16:31:42.840084Z","shell.execute_reply":"2022-11-08T16:31:42.846047Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.pipeline import Pipeline,make_pipeline



# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:31:44.038658Z","iopub.execute_input":"2022-11-08T16:31:44.039112Z","iopub.status.idle":"2022-11-08T16:31:44.062683Z","shell.execute_reply.started":"2022-11-08T16:31:44.039073Z","shell.execute_reply":"2022-11-08T16:31:44.061169Z"}}
df = pd.read_csv("../input/heart-disease-dataset/heart.csv")
df.head()
df.shape

# %% [code] {"jupyter":{"outputs_hidden":false},"_kg_hide-input":false,"execution":{"iopub.status.busy":"2022-11-08T16:31:44.789308Z","iopub.execute_input":"2022-11-08T16:31:44.789720Z","iopub.status.idle":"2022-11-08T16:31:45.035465Z","shell.execute_reply.started":"2022-11-08T16:31:44.789687Z","shell.execute_reply":"2022-11-08T16:31:45.034117Z"}}
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:31:47.099820Z","iopub.execute_input":"2022-11-08T16:31:47.100276Z","iopub.status.idle":"2022-11-08T16:31:47.157621Z","shell.execute_reply.started":"2022-11-08T16:31:47.100237Z","shell.execute_reply":"2022-11-08T16:31:47.156510Z"}}
df.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:31:49.840727Z","iopub.execute_input":"2022-11-08T16:31:49.841440Z","iopub.status.idle":"2022-11-08T16:31:49.850901Z","shell.execute_reply.started":"2022-11-08T16:31:49.841399Z","shell.execute_reply":"2022-11-08T16:31:49.849806Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:31:51.595009Z","iopub.execute_input":"2022-11-08T16:31:51.595536Z","iopub.status.idle":"2022-11-08T16:31:51.601649Z","shell.execute_reply.started":"2022-11-08T16:31:51.595490Z","shell.execute_reply":"2022-11-08T16:31:51.599993Z"}}
print(categorical_cols)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:31:52.465883Z","iopub.execute_input":"2022-11-08T16:31:52.466615Z","iopub.status.idle":"2022-11-08T16:31:52.474005Z","shell.execute_reply.started":"2022-11-08T16:31:52.466574Z","shell.execute_reply":"2022-11-08T16:31:52.472392Z"}}
#They are already encoded

categorical_cols.remove('target')
categorical_cols.remove('sex')
categorical_inds.remove(1)
categorical_inds.remove(13)


print(categorical_cols)
print(counting_cols)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:31:54.663837Z","iopub.execute_input":"2022-11-08T16:31:54.664564Z","iopub.status.idle":"2022-11-08T16:31:54.676379Z","shell.execute_reply.started":"2022-11-08T16:31:54.664519Z","shell.execute_reply":"2022-11-08T16:31:54.675342Z"}}
for i in categorical_cols:
    print(i)
    print(df[i].value_counts())
    print()

# %% [code] {"execution":{"iopub.status.busy":"2022-11-08T16:31:58.085204Z","iopub.execute_input":"2022-11-08T16:31:58.086057Z","iopub.status.idle":"2022-11-08T16:32:00.389442Z","shell.execute_reply.started":"2022-11-08T16:31:58.086015Z","shell.execute_reply":"2022-11-08T16:32:00.387860Z"}}
x = 1
plt.figure(figsize = (20,20))

for i in df.columns:
    plt.subplot(4,4,x)
    plt.boxplot(df[i])
    plt.title(i)
    x = x+1

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:32:55.739072Z","iopub.execute_input":"2022-11-08T16:32:55.740003Z","iopub.status.idle":"2022-11-08T16:32:55.751118Z","shell.execute_reply.started":"2022-11-08T16:32:55.739957Z","shell.execute_reply":"2022-11-08T16:32:55.749992Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=31)
X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:32:56.238866Z","iopub.execute_input":"2022-11-08T16:32:56.239616Z","iopub.status.idle":"2022-11-08T16:32:56.248766Z","shell.execute_reply.started":"2022-11-08T16:32:56.239560Z","shell.execute_reply":"2022-11-08T16:32:56.247433Z"}}
df.iloc[102,:-1].values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:32:56.918149Z","iopub.execute_input":"2022-11-08T16:32:56.918635Z","iopub.status.idle":"2022-11-08T16:32:56.925123Z","shell.execute_reply.started":"2022-11-08T16:32:56.918595Z","shell.execute_reply":"2022-11-08T16:32:56.923730Z"}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:32:58.513385Z","iopub.execute_input":"2022-11-08T16:32:58.513856Z","iopub.status.idle":"2022-11-08T16:32:58.520611Z","shell.execute_reply.started":"2022-11-08T16:32:58.513821Z","shell.execute_reply":"2022-11-08T16:32:58.519091Z"}}
from sklearn.preprocessing import RobustScaler
trf4 = make_column_transformer((RobustScaler(),counting_inds),remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:32:59.342516Z","iopub.execute_input":"2022-11-08T16:32:59.343157Z","iopub.status.idle":"2022-11-08T16:32:59.361550Z","shell.execute_reply.started":"2022-11-08T16:32:59.343115Z","shell.execute_reply":"2022-11-08T16:32:59.360232Z"}}
pipe = make_pipeline(trf3,trf4)


X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:01.032490Z","iopub.execute_input":"2022-11-08T16:33:01.032912Z","iopub.status.idle":"2022-11-08T16:33:01.042066Z","shell.execute_reply.started":"2022-11-08T16:33:01.032881Z","shell.execute_reply":"2022-11-08T16:33:01.040994Z"}}
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
    

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:03.023969Z","iopub.execute_input":"2022-11-08T16:33:03.024439Z","iopub.status.idle":"2022-11-08T16:33:03.436938Z","shell.execute_reply.started":"2022-11-08T16:33:03.024400Z","shell.execute_reply":"2022-11-08T16:33:03.435194Z"}}
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=2)
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:04.176302Z","iopub.execute_input":"2022-11-08T16:33:04.176749Z","iopub.status.idle":"2022-11-08T16:33:04.209449Z","shell.execute_reply.started":"2022-11-08T16:33:04.176716Z","shell.execute_reply":"2022-11-08T16:33:04.207784Z"}}
eval_metric(classifier1, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:06.793628Z","iopub.execute_input":"2022-11-08T16:33:06.794056Z","iopub.status.idle":"2022-11-08T16:33:06.804221Z","shell.execute_reply.started":"2022-11-08T16:33:06.794023Z","shell.execute_reply":"2022-11-08T16:33:06.803363Z"}}
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 2)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:09.007206Z","iopub.execute_input":"2022-11-08T16:33:09.008255Z","iopub.status.idle":"2022-11-08T16:33:09.026059Z","shell.execute_reply.started":"2022-11-08T16:33:09.008213Z","shell.execute_reply":"2022-11-08T16:33:09.024770Z"}}
eval_metric(classifier2, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:11.224946Z","iopub.execute_input":"2022-11-08T16:33:11.225732Z","iopub.status.idle":"2022-11-08T16:33:16.133982Z","shell.execute_reply.started":"2022-11-08T16:33:11.225679Z","shell.execute_reply":"2022-11-08T16:33:16.132471Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:16.136754Z","iopub.execute_input":"2022-11-08T16:33:16.138064Z","iopub.status.idle":"2022-11-08T16:33:16.366848Z","shell.execute_reply.started":"2022-11-08T16:33:16.138003Z","shell.execute_reply":"2022-11-08T16:33:16.365367Z"}}
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier()
classifier3.fit(X_train, y_train)
y_pred = classifier3.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:16.369470Z","iopub.execute_input":"2022-11-08T16:33:16.369902Z","iopub.status.idle":"2022-11-08T16:33:16.435112Z","shell.execute_reply.started":"2022-11-08T16:33:16.369856Z","shell.execute_reply":"2022-11-08T16:33:16.433775Z"}}
eval_metric(classifier3, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:23.497430Z","iopub.execute_input":"2022-11-08T16:33:23.497898Z","iopub.status.idle":"2022-11-08T16:33:23.540643Z","shell.execute_reply.started":"2022-11-08T16:33:23.497861Z","shell.execute_reply":"2022-11-08T16:33:23.538910Z"}}
from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 2)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:33.906582Z","iopub.execute_input":"2022-11-08T16:33:33.907409Z","iopub.status.idle":"2022-11-08T16:33:34.028306Z","shell.execute_reply.started":"2022-11-08T16:33:33.907367Z","shell.execute_reply":"2022-11-08T16:33:34.026586Z"}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:38.211619Z","iopub.execute_input":"2022-11-08T16:33:38.212089Z","iopub.status.idle":"2022-11-08T16:33:39.436838Z","shell.execute_reply.started":"2022-11-08T16:33:38.212049Z","shell.execute_reply":"2022-11-08T16:33:39.434857Z"}}
scores=[]

for i in range(1,40):
    classifier4 = KNeighborsClassifier(n_neighbors = i)
    classifier4.fit(X_train, y_train)
    y_pred = classifier4.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:39.440259Z","iopub.execute_input":"2022-11-08T16:33:39.441393Z","iopub.status.idle":"2022-11-08T16:33:39.452245Z","shell.execute_reply.started":"2022-11-08T16:33:39.441321Z","shell.execute_reply":"2022-11-08T16:33:39.450652Z"}}
maxi_ind=-1
maxi=0
cnt=0
for i in scores:
    cnt+=1
    if maxi<i:
        max_ind = cnt
        maxi = i
print(max_ind)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:39.785050Z","iopub.execute_input":"2022-11-08T16:33:39.785473Z","iopub.status.idle":"2022-11-08T16:33:39.824894Z","shell.execute_reply.started":"2022-11-08T16:33:39.785441Z","shell.execute_reply":"2022-11-08T16:33:39.822901Z"}}
classifier4 = KNeighborsClassifier(n_neighbors = max_ind)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:42.685929Z","iopub.execute_input":"2022-11-08T16:33:42.686849Z","iopub.status.idle":"2022-11-08T16:33:42.798067Z","shell.execute_reply.started":"2022-11-08T16:33:42.686795Z","shell.execute_reply":"2022-11-08T16:33:42.796806Z"}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:58.276733Z","iopub.execute_input":"2022-11-08T16:33:58.277155Z","iopub.status.idle":"2022-11-08T16:33:58.288290Z","shell.execute_reply.started":"2022-11-08T16:33:58.277121Z","shell.execute_reply":"2022-11-08T16:33:58.287125Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=31)

X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:33:59.231658Z","iopub.execute_input":"2022-11-08T16:33:59.232323Z","iopub.status.idle":"2022-11-08T16:33:59.237336Z","shell.execute_reply.started":"2022-11-08T16:33:59.232274Z","shell.execute_reply":"2022-11-08T16:33:59.236439Z"}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:00.856509Z","iopub.execute_input":"2022-11-08T16:34:00.856972Z","iopub.status.idle":"2022-11-08T16:34:00.862657Z","shell.execute_reply.started":"2022-11-08T16:34:00.856932Z","shell.execute_reply":"2022-11-08T16:34:00.861802Z"}}
from sklearn.preprocessing import RobustScaler
trf4 = make_column_transformer((RobustScaler(),counting_inds),remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:01.662010Z","iopub.execute_input":"2022-11-08T16:34:01.662812Z","iopub.status.idle":"2022-11-08T16:34:01.672499Z","shell.execute_reply.started":"2022-11-08T16:34:01.662764Z","shell.execute_reply":"2022-11-08T16:34:01.671355Z"}}
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:03.182270Z","iopub.execute_input":"2022-11-08T16:34:03.183109Z","iopub.status.idle":"2022-11-08T16:34:03.215568Z","shell.execute_reply.started":"2022-11-08T16:34:03.183063Z","shell.execute_reply":"2022-11-08T16:34:03.214379Z"}}
pipe = make_pipeline(trf3,trf4,classifier)


pipe.fit(X_train,y_train)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:05.613874Z","iopub.execute_input":"2022-11-08T16:34:05.614340Z","iopub.status.idle":"2022-11-08T16:34:05.628500Z","shell.execute_reply.started":"2022-11-08T16:34:05.614261Z","shell.execute_reply":"2022-11-08T16:34:05.627077Z"}}
y_pred = pipe.predict(X_test)
y_pred

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:09.216126Z","iopub.execute_input":"2022-11-08T16:34:09.217054Z","iopub.status.idle":"2022-11-08T16:34:09.242883Z","shell.execute_reply.started":"2022-11-08T16:34:09.216999Z","shell.execute_reply":"2022-11-08T16:34:09.241749Z"}}
eval_metric(pipe, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:13.695221Z","iopub.execute_input":"2022-11-08T16:34:13.695688Z","iopub.status.idle":"2022-11-08T16:34:13.701844Z","shell.execute_reply.started":"2022-11-08T16:34:13.695649Z","shell.execute_reply":"2022-11-08T16:34:13.700914Z"}}
import pickle

pickle.dump(pipe,open('pipe.pkl','wb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:14.497691Z","iopub.execute_input":"2022-11-08T16:34:14.498939Z","iopub.status.idle":"2022-11-08T16:34:14.505761Z","shell.execute_reply.started":"2022-11-08T16:34:14.498878Z","shell.execute_reply":"2022-11-08T16:34:14.504540Z"}}
pipe = pickle.load(open('pipe.pkl','rb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:15.211137Z","iopub.execute_input":"2022-11-08T16:34:15.211943Z","iopub.status.idle":"2022-11-08T16:34:15.219364Z","shell.execute_reply.started":"2022-11-08T16:34:15.211901Z","shell.execute_reply":"2022-11-08T16:34:15.218313Z"}}
test1 = np.array([53,1,0,140,203,1,0,155,1,3.1,0,0,3]).reshape(1,13)

# %% [code] {"execution":{"iopub.status.busy":"2022-11-08T16:34:17.261829Z","iopub.execute_input":"2022-11-08T16:34:17.262251Z","iopub.status.idle":"2022-11-08T16:34:17.267758Z","shell.execute_reply.started":"2022-11-08T16:34:17.262218Z","shell.execute_reply":"2022-11-08T16:34:17.266823Z"}}
test2 = np.array([45,1,2,255,255,1,2,255,1,1.4,2,4,3]).reshape(1,13)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:17.887378Z","iopub.execute_input":"2022-11-08T16:34:17.888128Z","iopub.status.idle":"2022-11-08T16:34:17.897362Z","shell.execute_reply.started":"2022-11-08T16:34:17.888086Z","shell.execute_reply":"2022-11-08T16:34:17.896504Z"}}
pipe.predict(test1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-08T16:34:19.504458Z","iopub.execute_input":"2022-11-08T16:34:19.505355Z","iopub.status.idle":"2022-11-08T16:34:19.516246Z","shell.execute_reply.started":"2022-11-08T16:34:19.505173Z","shell.execute_reply":"2022-11-08T16:34:19.514964Z"}}
pipe.predict(df.iloc[102,:-1].values.reshape(1,13))

# %% [code]
