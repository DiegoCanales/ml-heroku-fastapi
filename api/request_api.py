import requests

data_test_zero = {'workclass': 'State-gov', 'education': 'HS-grad', 'marital-status': 'Divorced', 'occupation': 'Adm-clerical',
            'race': 'White', 'sex': 'Female', 'native-country': 'United-States', 'age': 50, 'hours-per-week': 46, 'salary': '<=50K'}
label_zero = data_test_zero.pop('salary')

data_test_one = {'workclass': 'Private', 'education': 'Bachelors', 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
            'race': 'White', 'sex': 'Male', 'native-country': 'United-States', 'age': 49, 'hours-per-week': 66, 'salary': '>50K'}
label_one = data_test_one.pop('salary')

print("Request 1")
response = requests.post('https://census-mlops.herokuapp.com/inference',
                         json=data_test_zero)
print(f"Status Code: {response.status_code}")
print(f"Data testing: {data_test_zero}")
print(f"Ground Truth: {label_zero}, Response: {response.json()}")

print("=============================================")

print("Request 2")
response = requests.post('https://census-mlops.herokuapp.com/inference',
                         json=data_test_one)
print(f"Status Code: {response.status_code}")
print(f"Data testing: {data_test_one}")
print(f"Ground Truth: {label_one}, Response: {response.json()}")
