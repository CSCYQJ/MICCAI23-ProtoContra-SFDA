import os
import numpy as np
import torch
import time


class IterationCounter():
    def __init__(self, opt):
        self.opt = opt
        self.steps_so_far = 0
        self.epochs_so_far = 0
        self.time_measurements = {}
        self.batch_size = opt['batch_size']

    def record_one_iteration(self):
        self.steps_so_far += 1
    
    def record_one_epoch(self):
        self.epochs_so_far +=1

    def needs_saving(self):
        return ((self.epochs_so_far % self.opt['save_epochs']) == 0) and self.epochs_so_far > 0
    
    def needs_saving_steps(self):
        return ((self.steps_so_far % self.opt['save_steps']) == 0) and self.steps_so_far > 0

    def needs_evaluation(self):
        return (self.epochs_so_far >= self.opt['eval_epochs']) and \
            ((self.epochs_so_far % self.opt['eval_epochs']) == 0 )
    
    def needs_evaluation_steps(self):
        return (self.steps_so_far >= self.opt['eval_steps']) and \
            ((self.steps_so_far % self.opt['eval_steps']) == 0 )

    def needs_displaying(self):
        return (self.steps_so_far % self.opt['display_steps']) == 0


    class TimeMeasurement:
        def __init__(self, name, parent):
            self.name = name
            self.parent = parent

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, type, value, traceback):
            torch.cuda.synchronize(device = self.parent.opt['gpu_id'])
            start_time = self.start_time
            elapsed_time = (time.time() - start_time) / self.parent.batch_size

            if self.name not in self.parent.time_measurements:
                self.parent.time_measurements[self.name] = elapsed_time
            else:
                prev_time = self.parent.time_measurements[self.name]
                updated_time = prev_time * 0.98 + elapsed_time * 0.02
                self.parent.time_measurements[self.name] = updated_time

    def time_measurement(self, name):
        return IterationCounter.TimeMeasurement(name, self)

