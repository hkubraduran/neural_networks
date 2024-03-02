import math
import numpy as np
"""
softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]
loss = -(math.log(softmax_output[0])* target_output[0] +
         math.log(softmax_output[1])* target_output[1] +
         math.log(softmax_output[2])* target_output[2])

print(loss)

loss = -(math.log(softmax_output[0]))
print(loss)
print(-math.log(0.7))
print(-math.log(0.5))"""

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]
class_targets = [0, 1, 1]

for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]])

print(softmax_outputs[[0, 1, 2], class_targets])

print(softmax_outputs[range(len(softmax_outputs)), class_targets])

# to calculate the loss
# equals to negative log of the target class' confidence
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(neg_log)

avg_loss = np.mean(neg_log)
print(avg_loss)

# RuntimeWarning: divide by zero encountered in log
# print(-np.log(0))

softmax_outputs = np.array([
    [0.0, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]])
# print(-np.log(softmax_outputs[[0, 1, 2], class_targets]))
# the first loss will be infinite

# print(np.mean(-np.log(softmax_outputs[[0, 1, 2], class_targets])))
# RuntimeWarning: divide by zero encountered in log

"""One option that we have to handle this is 
to clip the values in the range by some fairly insignificant amount"""

print(-np.log(1-1e-7))

# y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
