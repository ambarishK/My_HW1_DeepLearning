#!/usr/bin/env python
# coding: utf-8

# # **CSCE 5218 / CSCE 4930 Deep Learning**
# 
# # **HW1a The Perceptron** (20 pt)
# 

# In[1]:


# Get the datasets
get_ipython().system('wget http://huang.eng.unt.edu/CSCE-5218/test.dat')
get_ipython().system('wget http://huang.eng.unt.edu/CSCE-5218/train.dat')


# In[2]:


# Take a peek at the datasets
get_ipython().system('head train.dat')
get_ipython().system('head test.dat')


# ### Build the Perceptron Model
# 
# You will need to complete some of the function definitions below.  DO NOT import any other libraries to complete this. 

# In[16]:


import math
import itertools
import re


# Corpus reader, all columns but the last one are coordinates;
#   the last column is the label
def read_data(file_name):
    f = open(file_name, 'r')

    data = []
    # Discard header line
    f.readline()
    for instance in f.readlines():
        if not re.search('\t', instance): continue
        instance = list(map(int, instance.strip().split('\t')))
        # Add a dummy input so that w0 becomes the bias
        instance = [-1] + instance
        data += [instance]
    return data


def dot_product(weight, instance):
    dot_product=sum([x*y for(x,y) in zip(weight,instance)])
    return dot_product 


def sigmoid(x):
    sigmd = 1/(1 +math.exp(-x))
    return sigmd

# The output of the model, which for the perceptron is 
# the sigmoid function applied to the dot product of 
# the instance and the weights
def output(weight, instance):
    outpt = sigmoid(dot_product(weight,instance))
    return outpt

# Predict the label of an instance; this is the definition of the perceptron
# you should output 1 if the output is >= 0.5 else output 0
def predict(weight, instance):
    if output(weight,instance) >= 0.5:
        return 1
    else:
        return 0


# Accuracy = percent of correct predictions
def get_accuracy(weights, instances):
    # You do not to write code like this, but get used to it
    correct = sum([1 if predict(weights, instance) == instance[-1] else 0
                   for instance in instances])
    return correct * 100 / len(instances)


# Train a perceptron with instances and hyperparameters:
#       lr (learning rate) 
#       epochs
# The implementation comes from the definition of the perceptron
#
# Training consists on fitting the parameters which are the weights
# that's the only thing training is responsible to fit
# (recall that w0 is the bias, and w1..wn are the weights for each coordinate)
#
# Hyperparameters (lr and epochs) are given to the training algorithm
# We are updating weights in the opposite direction of the gradient of the error,
# so with a "decent" lr we are guaranteed to reduce the error after each iteration.
def train_perceptron(instances, lr, epochs):

    #TODO: name this step
    weights = [0] * (len(instances[0])-1)

    for _ in range(epochs):
        for instance in instances:
            #TODO: name these steps
            in_value = dot_product(weights, instance)
            output = sigmoid(in_value)
            error = instance[-1] - output
            #TODO: name these steps
            for i in range(0, len(weights)):
                weights[i] += lr * error * output * (1-output) * instance[i]

    return weights


# ## Run it

# In[17]:


instances_tr = read_data("train.dat")
instances_te = read_data("test.dat")
lr = 0.005
epochs = 5
weights = train_perceptron(instances_tr, lr, epochs)
accuracy = get_accuracy(weights, instances_te)
print(f"#tr: {len(instances_tr):3}, epochs: {epochs:3}, learning rate: {lr:.3f}; "
      f"Accuracy (test, {len(instances_te)} instances): {accuracy:.1f}")


# ## Questions
# 
# Answer the following questions. Include your implementation and the output for each question.

# 
# 
# ### Question 1
# 
# In `train_perceptron(instances, lr, epochs)`, we have the follosing code:
# ```
# in_value = dot_product(weights, instance)
# output = sigmoid(in_value)
# error = instance[-1] - output
# ```
# 
# Why don't we have the following code snippet instead?
# ```
# output = predict(weights, instance)
# error = instance[-1] - output
# ```
# 
# #### TODO Add your answer here (text only)
# Topmost code snippet is application of sigmoid activation function. Which enables the fine tunned perceptron model to classify the data with enhanced accuracy and precision. Sigmoid activation function assumes and accepts any real value between 0 and 1 and based on the set threshold performes the classification. Here, usage of sigmoid activation function is aimed to achieve a perceptron model with far better performance in terms of accuracy and precision. In other words non-linear activation function outperformes binary or linear activation function. Linear activation function or binary step function only accepts either 0 or 1 not any other real number between 0 and 1.This retards the performance of model using linear or binary step function.  
# 
# 
# 

# ### Question 2
# Train the perceptron with the following hyperparameters and calculate the accuracy with the test dataset.
# 
# ```
# tr_percent = [5, 10, 25, 50, 75, 100] # percent of the training dataset to train with
# num_epochs = [5, 10, 20, 50, 100]              # number of epochs
# lr = [0.005, 0.01, 0.05]              # learning rate
# ```
# 
# TODO: Write your code below and include the output at the end of each training loop (NOT AFTER EACH EPOCH)
# of your code.The output should look like the following:
# ```
# # tr:  20, epochs:   5, learning rate: 0.005; Accuracy (test, 100 instances): 68.0
# # tr:  20, epochs:  10, learning rate: 0.005; Accuracy (test, 100 instances): 68.0
# # tr:  20, epochs:  20, learning rate: 0.005; Accuracy (test, 100 instances): 68.0
# [and so on for all the combinations]
# ```
# You will get different results with different hyperparameters.
# 
# #### TODO Add your answer here (code and output in the format above) 
# 

# In[25]:


instances_tr = read_data("train.dat")
instances_te = read_data("test.dat")
tr_percent = [5, 10, 25, 50, 75, 100] # percent of the training dataset to train with
num_epochs = [5, 10, 20, 50, 100]     # number of epochs
lr_array = [0.005, 0.01, 0.05]        # learning rate
accur=[]
epoc=[]
tr_set=[]
lr_rate=[]

for lr in lr_array:
  for tr_size in tr_percent:
    for epochs in num_epochs:
      size =  round(len(instances_tr)*tr_size/100)
      pre_instances = instances_tr[0:size]
      weights = train_perceptron(pre_instances, lr, epochs)
      accuracy = get_accuracy(weights, instances_te)
      
    print(f"#tr: {len(pre_instances):0}, epochs: {epochs:3}, learning rate: {lr:.3f}; "
            f"Accuracy (test, {len(instances_te)} instances): {accuracy:.1f}")
    accur.append(accuracy)
    epoc.append(epochs)
    lr_rate.append(lr)   
    tr_set.append(len(pre_instances))


# ### Question 3
# Write a couple paragraphs interpreting the results with all the combinations of hyperparameters. Drawing a plot will probably help you make a point. In particular, answer the following:
# 
# Above results are obtained by using different learning parameters - percent of training set, epochs and learning rate. Accuracy is varied for different combinations of parameters. Lower accuracy is achived with lesser percent of training dataset i.e 5 percent of training dataset. Increasing the used percent of training dataset increases the accuracy. Apart, learning rate has significant role in performance of the model. Given the percent of training dataset, lower learning rate has lower accuracy whereas higher learning rate has high accuracy. Number of epoch also decides the accuracy. More the number of epochs, it better updates the weights and help achieve the local minima. So, high number of epochs with other given parameters has better performance. 
# - A. Do you need to train with all the training dataset to get the highest accuracy with the test dataset?
# Yes, we need to train with all the training datset to get the highest accuracy with the test dataset.
# - B. How do you justify that training the second run obtains worse accuracy than the first one (despite the second one uses more training data)?
# 
# 
#  
#    ```
# #tr: 100, epochs:  20, learning rate: 0.050; Accuracy (test, 100 instances): 71.0
# #tr: 200, epochs:  20, learning rate: 0.005; Accuracy (test, 100 instances): 68.0
# 
# In second run, inspite of more percent of training dataset learning rate is lower i.e 0.005 which does not let the model to converge to minima. In other words, weights are not updated/learned to reach to the minima. So, in second run, we are not getting best updated weights and hence accuracy is lower.  
# ```
# - C. Can you get higher accuracy with additional hyperparameters (higher than `80.0`)?
# 
# Yes, there is likely that we can get higher accuracy with additional hyperparameters. Additional hyperparameters can be increased number of hidden layers, increased number of neurons etc.
# 
# - D. Is it always worth training for more epochs (while keeping all other hyperparameters fixed)?
# 
# No, it is not always worth training for more epochs keeping the other hyperparametrs fixed. It is likely the lesser number of epochs will yield best accuracy and increasing the number of epochs will not impact on performance. 
# 
# 
# #### TODO: Add your answer here (code and text)
# 
# 

# In[26]:


print (accur)


# 

# In[27]:


import matplotlib.pyplot as plt 


# In[28]:


plt.scatter(epoc, accur, c='r', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting primes')
plt.show()


# In[29]:


plt.scatter(tr_set, accur, c='r', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting primes')
plt.show()


# In[30]:


plt.scatter(lr_rate, accur, c='r', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting primes')
plt.show()


# In[ ]:




