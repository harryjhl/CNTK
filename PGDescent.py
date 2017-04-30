from cntk.learners import UserLearner
import math
import numpy as np

class PGDescent(UserLearner):

    def __init__(self, parameters, lr_schedule, l1_regularization_weight=0.0001, l2_regularization_weight=0.001):
        super(PGDescent, self).__init__(parameters, lr_schedule)
        self.l1 = l1_regularization_weight
        self.l2 = l2_regularization_weight
        self.count = 0

    def update(self, gradient_values, training_sample_count, sweep_end):

        l1 = self.l1
        l2 = self.l2
        eta = self.learning_rate()/training_sample_count

        for p in gradient_values:
            p.value -= gradient_values[p].to_ndarray()*eta
            p.value = np.sign(p.value) * np.maximum(np.abs(p.value)-eta*l2, 0.0) / (1.0+l2*eta)

        self.count += 1
        return True