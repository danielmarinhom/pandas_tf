from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import pandas

file = pandas.read_csv('/home/kali/Desktop/teste.csv')


#x = nome, cpf
#y = sexo
X = file.drop(columns=['Sexo'])
Y = file['Sexo'].astype('float32').values


model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

model.fit(X, Y, epochs=200, batch_size=32)

y_hat = model.predict(X)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

print(accuracy_score(Y, y_hat))

model.save('tfmodel')


#loading model
model = load_model('tfmodel')