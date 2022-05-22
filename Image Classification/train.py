from preprocessing import Preprocessing
from model import Model
data_path_train = "G:/rauf/STEPBYSTEP/Data2/Custom/horse vs zebra/train"
data_path_test = "G:/rauf/STEPBYSTEP/Data2/Custom/horse vs zebra/test"

data_loader = Preprocessing()

# train data
X_train, y_train = data_loader.load_train_data(data_path_train)
y_train = data_loader.convert_name(y_train)

# test data
X_test, y_test = data_loader.load_train_data(data_path_test)
y_test = data_loader.convert_name(y_train)
print(len(y_train), len(y_test))

model = Model()
model = model.create_model('basic')
model.fit(X_train, y_train, epochs=10)