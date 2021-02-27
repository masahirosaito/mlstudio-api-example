import requests
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix


scoring_uri = 'http://3f682bdc-98bc-4078-a147-2689675a7ff1.eastus.azurecontainer.io/score'
key = 'Gf4I7WHEciwssUPHuG3r1K3Xu941b6lq'


def csv_to_list(csv_path):
    df = pd.read_csv(csv_path)
    features_list = df.drop('Band_gap', axis=1).values.tolist()
    target_list = df['Band_gap'].values.tolist()
    
    return target_list, features_list


def get_predict(features_list, uri, key):
    features_dict = {"data": features_list}
    input_data = json.dumps(features_dict)
    headers = {'Content-Type': 'application/json'}
    headers['Authorization'] = f'Bearer {key}' 
    resp = requests.post(uri, input_data, headers=headers)
    result = resp.text
    predict_dict=json.loads(json.loads(result))
    return predict_dict["result"]


test_target, test_features = csv_to_list('BG_test.csv')
test_predict = get_predict(test_features, scoring_uri, key)


train_target, train_features = csv_to_list('BG_train.csv')
train_predict = get_predict(train_features, scoring_uri, key)


print(len(test_predict))
print(len(train_target))


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(train_target, train_predict, label='Train set')
ax.scatter(test_target, test_predict, label='Test set')
ax.set_aspect('equal') 
x = np.arange(0, 9, 0.1)
y = x
plt.plot(x, y, c="black", linestyle="dashed")
plt.xlim(0,9)
plt.ylim(0,9)
plt.xlabel("True")
plt.ylabel("Predict")
plt.legend()
plt.show()


def get_material_data(mpID):
    with MPRester(material_key, notify_db_version=False) as m:
        try:
            data = m.get_data(mpID)
        except Exception:
            print("error : get_ipython().run_line_magic("s"", " % mpID)")
            exit()
    return data


true_test = [ (i > 2) & (i < 3) for i in test_target]
predict_test = [ (i > 2) & (i < 3) for i in test_predict]
positive=[]
f_negative=[]
for true, predict, features, bandgap in zip(true_test, predict_test, test_features, test_target):
    if predict:
        if true:
            correctness='〇'
        else:
            correctness='×'
        positive.append([features[0], features[1], features[2], bandgap, correctness])
    elif true and not(predict):
        f_negative.append([features[0], features[1], features[2], bandgap])
positive_df = pd.DataFrame(positive, columns=['ID', 'Formula', 'Space group', 'Band gap', 'Correctness'])
f_negative_df = pd.DataFrame(f_negative, columns=['ID', 'Formula', 'Space group', 'Band gap'])


positive_df


f_negative_df


cm_test = confusion_matrix(true_test, predict_test)
sns.heatmap(cm_test, annot=True, cmap='Blues')
plt.show()



