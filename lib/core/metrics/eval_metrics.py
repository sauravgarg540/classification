import torch


class Metrics:
    def __init__(self, num_classes=None, epsilon=1e-10):
        self.num_classes = num_classes
        self.num_samples = 0
        self.epsilon = epsilon
        self.reset()

    @torch.no_grad()
    def update(self, output, target):
        self.num_samples += target.size(0)

        if output.dim() > 1:
            _, output = output.topk(1, 1, True, True)
        self.output.append(output.view(-1))
        self.target.append(target)

    def reset(self):
        self.output = []
        self.target = []

    @torch.no_grad()
    def _process(self):
        self.output = torch.cat(self.output, 0)
        self.target = torch.cat(self.target, 0)
        tp = torch.empty(self.num_classes)
        fp = torch.empty(self.num_classes)
        fn = torch.empty(self.num_classes)
        tn = torch.empty(self.num_classes)
        for i in range(self.num_classes):
            tp[i] = ((self.output == i) & (self.target == i)).sum().item()
            fp[i] = ((self.output == i) & (self.target != i)).sum().item()
            fn[i] = ((self.output != i) & (self.target == i)).sum().item()
            tn[i] = ((self.output != i) & (self.target != i)).sum().item()
        return torch.sum(tp), torch.sum(fp), torch.sum(fn), torch.sum(tn)

    @torch.no_grad()
    def compute_metrics(self):
        tp, fp, fn, tn = self._process()
        tp = tp.float()
        fp = fp.float()
        fn = fn.float()
        tn = tn.float()
        accuracy = (tp + tn) / (tp + fp + fn + tn + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        precision = tp / (tp + fp + self.epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + self.epsilon)

        matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[j][i] = ((self.output == i) & (self.target == j)).sum().item()
        return accuracy, recall, precision, f1_score, matrix
