class InvScheduler(object):
    def __init__(self, gamma, decay_rate, group_ratios, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.group_ratios = group_ratios
        self.init_lr = init_lr

    def adjust_learning_rate(self, optimizer, iteration):
        lr = self.init_lr * (1 + self.gamma * iteration) ** (-self.decay_rate)
        for (param_group, ratio) in zip(optimizer.param_groups, self.group_ratios):
            param_group['lr'] = lr * ratio
