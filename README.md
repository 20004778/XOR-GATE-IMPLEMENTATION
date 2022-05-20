### EX NO : 08
### DATE  : 13.05.2022
# <p align="center"> XOR GATE IMPLEMENTATION </p>
## Aim:
   To implement multi layer artificial neural network using back propagation algorithm.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## Related Theory Concept:
logic gates using neural networks help understand the mathematical computation by which a neural network processes its inputs to arrive at a certain output. This neural network will deal with the XOR logic problem. An XOR (exclusive OR gate) is a digital logic gate that gives a true output only when both its inputs differ from each other.

The information of a neural network is stored in the interconnections between the neurons i.e. the weights. A neural network learns by updating its weights according to a learning algorithm that helps it converge to the expected output. The learning algorithm is a principled way of changing the weights and biases based on the loss function.

## Algorithm
1.Import necessary packages\
2.Set the four different states of the XOR gate\
3.Set the four expected results in the same order\
4.Get the accuracy\
5.Train the model with training data.\
6.Now test the model with testing data.


## Program:
```python
Program to implement XOR Logic Gate.
Developed by   : SURYA R
RegisterNumber : 212220230052
```

# XOR Logic gate implementation using ANN
```python
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data =  np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")

model =Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print(model.predict(training_data).round())
```

## Output:

![s](https://user-images.githubusercontent.com/75236145/169481590-48f16c51-a4e0-4118-ad67-46d11e9e9368.png)
![r](https://user-images.githubusercontent.com/75236145/169481633-b5da0632-3d16-4f2d-90b7-f53a82a3658d.png)


## Result:
Thus the python program successully implemented XOR logic gate.
