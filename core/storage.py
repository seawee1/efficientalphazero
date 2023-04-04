import ray


def add_logs(src_dict, target_dict):
    for k, v in src_dict.items():
        if k not in target_dict:
            target_dict[k] = []
        if isinstance(v, list):
            target_dict[k].extend(v)
        else:
            target_dict[k].append(v)


@ray.remote
class SharedStorage:
    def __init__(self, config, amp):
        self.config = config
        self.model = config.create_model('cpu', amp)

        self.start = False
        self.step_counter = 0

        self.rollout_worker_log = {}
        self.test_worker_log = {}
        self.test_train_step = -1
        self.workers_finished = 0

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def incr_workers_finished(self):
        self.workers_finished += 1

    def reset_workers_finished(self):
        self.workers_finished = 0

    def get_workers_finished(self):
        return self.workers_finished

    def add_rollout_worker_logs(self, log_dict):
        add_logs(log_dict, self.rollout_worker_log)

    def pop_rollout_worker_logs(self):
        logs = self.rollout_worker_log
        self.rollout_worker_log = {}
        return logs
