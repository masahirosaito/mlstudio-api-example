import requests
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix


from pymatgen import MPRester


scoring_uri = 'http://0af890ef-0e8a-4e83-b0bb-6263db52e790.eastus.azurecontainer.io/score'
key = 'Vv2F1vu4al7e6pOlr1TzAfZBfoSBFn2v'


material_key='I3mgNiZBnTJEtSa7'


def csv_to_list(csv_path):
    df = pd.read_csv(csv_path)
    ID_list = df.iloc[:,2]
    features_list = df.values.tolist()
    target_list = df.iloc[:,0].values.tolist()
    
    return ID_list, target_list, features_list


def get_predict(features_list, uri, key):
    features_dict = {"data": features_list}
    input_data = json.dumps(features_dict)
    headers = {'Content-Type': 'application/json'}
    headers['Authorization'] = f'Bearer {key}' 
    resp = requests.post(uri, input_data, headers=headers)
    result = resp.text
    predict_dict=json.loads(json.loads(result))
    return predict_dict["result"]


test_ID, test_target, test_features = csv_to_list('BG_test.csv')
test_predict = get_predict(test_features, scoring_uri, key)


train_ID, train_target, train_features = csv_to_list('BG_train.csv')
train_predict = get_predict(train_features, scoring_uri, key)


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
for mpID, true, predict, bandgap in zip(test_ID, true_test, predict_test, test_target):
    if predict:
        data = get_material_data(mpID)
        if true:
            correctness='〇'
        else:
            correctness='×'
        positive.append([mpID, data[0]['pretty_formula'], data[0]['spacegroup']['symbol'], bandgap, correctness])
    elif true and not(predict):
        data = get_material_data(mpID)
        f_negative.append([mpID, data[0]['pretty_formula'], data[0]['spacegroup']['symbol'], bandgap])
positive_df = pd.DataFrame(positive, columns=['ID', 'Formula', 'Space group', 'Band gap', 'Correctness'])
f_negative_df = pd.DataFrame(f_negative, columns=['ID', 'Formula', 'Space group', 'Band gap'])


positive_df


f_negative_df


cm_test = confusion_matrix(true_test, predict_test)
sns.heatmap(cm_test, annot=True, cmap='Blues')
plt.show()



