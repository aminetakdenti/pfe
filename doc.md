# Data Laoder

First we need to laod the data to be used on the DQN model
so for that according to the paper 
we need devide our data to small mini batches
and on each mini batch there is 3 stuff

- The state (st)
- The action (at)
- The next state (st+1)

the st and st+1 are independent so there is no relation between them

and the way we select the batches are random
so we can use the random library to select the batches
and also to select the next state (st+1)

# DQN Model

## 1. Predict MODEL
it's model that take 3 layers with the Leru on the output of each layer 

the first layer is the size of the features of (st)

and the output will be the size of the action space

## 2. Traning MODEL
the model is the same as the predict model but with a different loss function
and the loss function is the mean square error
and the optimizer is the Adam optimizer
and the learning rate is 0.001

# Implementation

first we start by choosing the st, at, st+1
and then pass the st to the predict model and get the Q value, then we call the select action function that will select action based on the epsilon greedy policy
by that will have two new values
the action (at*) and the reward (r)
and the reward will 1/0 according to the predicted action if equal to the at then r will be 1 else 0

after that we pass the st+1 to the predict model and get the Q value and get the max Q value by calling the select action function with epsilon = 1 
after that we calc the target Q value using the Bellman equation
Qtar = r(REWARD) + lambda(DISCOUNT FACTORY) * max(Q(st+1))

after that we pass the st and qt* to the training model as input and calc the loss function the loss function is the mean square error of the Q value of the traning and the target Q value
and then we call the optimizer to update the weights of the model
