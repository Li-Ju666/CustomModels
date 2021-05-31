from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler
import joblib
# from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, LSTM

def data_read(filename):
    return read_csv(filename, header=None).values.astype("float32")

def series_to_supervised(data, n_in=1, n_out=1):
    agg = []
    for i in range(data.shape[0] - n_in - n_out + 1):
        agg.append(list(data[i: i+n_in+n_out, :].flatten()))
    agg = DataFrame(agg)
    return agg.values

def formulate(data, n_feature):
    data_x, data_y = data[:, :n_feature], data[:, n_feature:]
    return data_x.reshape(data_x.shape[0], 1, data_x.shape[1]), data_y

def init_model(n_feature):
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, n_feature)))
    model.add(Dense(n_feature))
    model.compile(loss='mae', optimizer='adam')
    return model

def train_model(model, train_set, val_set, epoch=1, batch_size=30):
    train_x, train_y = train_set
    model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,
              validation_data=val_set, verbose=2, shuffle=False)

train_samples = 2000
n_feature = 5

data = data_read("./metrics_log.csv")
train = data[0:train_samples, :]
test = data[train_samples:, :]

scaler = MinMaxScaler(feature_range=(0, 1))

train = series_to_supervised(scaler.fit_transform(train))
test = series_to_supervised(scaler.transform(test))

train_x, train_y = formulate(train, 5)
test_x, test_y = formulate(test, 5)

model = init_model(n_feature)
train_model(model, (train_x, train_y), (test_x, test_y), epoch=50)

# predict = scaler.inverse_transform(model.predict(test_x))
# test_y = scaler.inverse_transform(test_y)
#
# fig, axs = pyplot.subplots(5)
# for i in range(5):
#     axs[i].plot(test_y[:, i])
#     axs[i].plot(predict[:, i])
#
# fig.show()

model.save("./model.h5")
joblib.dump(scaler, "./scaler")

# newscaler = joblib.load("project/customAutoscaler/predictive-cpa/init/scaler.joblib")
