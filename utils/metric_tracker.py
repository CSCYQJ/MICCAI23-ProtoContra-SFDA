class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def moving_average(self, old, new):
        s = 0.98
        return old * (s) + new * (1 - s)

    def update_metrics(self, metric_dict, smoothe=True):
        for k, v in metric_dict.items():
            if k in self.metrics and smoothe:
                self.metrics[k] = self.moving_average(self.metrics[k], v)
            else:
                self.metrics[k] = v

    def current_metrics(self):
        return self.metrics
