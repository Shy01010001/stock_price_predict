import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask): # [batch_size, max_seq_len-1, vocab_size+1], [batch_size, max_seq_len-1]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask # [batch_size, max_seq_len-1]
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(output, reports_ids, reports_masks): # [batch_size, max_seq_len-1, vocab_size+1], [batch_size, max_seq_len]
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y, y_hat):
        residual = torch.abs(y - y_hat)
        loss = torch.where(residual < self.delta, 0.5 * residual.pow(2), self.delta * residual - 0.5 * self.delta ** 2)
        return loss.mean()