
import math
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
print(f"Exponential values: \n{exp_values}")

# print(f"sum without axis : \n{np.sum(layer_outputs)}")    #it gives us 1 value by sum all values
# print(f"sum with axis=0 :\n{np.sum(layer_outputs, axis=0)}")    #it gives us 3 values by sum the COLUMNS
# print(f"sum with axis=0 and transpose : \n{np.sum(np.array(layer_outputs).T, axis=0)}")
# #it gives us 3 values by sum the COLUMNS and this is what we want
print(f"sum with axis=1 : \n{np.sum(layer_outputs, axis=1)}")    #This is the better version
print(f"sum with axis=1 and shaping for division to normalize : \n{np.sum(layer_outputs, axis=1, keepdims=True)}")
"""This is the better version for divison to normalize the values 
It is a must-have to prevent shape error
"""

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(f"Normalized values: \n{norm_values}")

#print(exp_values)
