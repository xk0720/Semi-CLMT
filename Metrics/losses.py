import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score


def distill_loss(feature_stu, feature_tea, norm=False):

    if not norm:
        B, N, C = feature_stu.shape
        feature_stu = feature_stu.view(B, N, 1, C)
        feature_stu = feature_stu.repeat(1, 1, N, 1)
        feature_stu = feature_stu.view(B, N * N, C)

        feature_tea = feature_tea.repeat(1, N, 1)
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        score = cosine_similarity(feature_stu, feature_tea)  # B, N*N
        score = torch.exp(score.view(-1, N, N))
        mask = torch.eye(score.shape[1], dtype=torch.bool).cuda()
        distillation_loss = - torch.mean(torch.log(score[:, mask] / score.sum(dim=-1)).sum(dim=-1))

    else:
        score = torch.exp((feature_stu @ feature_tea.transpose(-2, -1)))
        mask = torch.eye(score.shape[1], dtype=torch.bool).cuda()
        distillation_loss = - torch.mean(torch.log(score[:, mask] / score.sum(dim=-1)).sum(dim=-1))

    return distillation_loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        dice_loss = self._dice_loss(inputs, target)
        loss = 1 - dice_loss
        return loss




def logits_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes logits on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean_teacher.
    - Sends gradients to inputs but not the targets.
    """

    assert input_logits.size() == target_logits.size()
    mse_loss = (input_logits-target_logits)**2
    return mse_loss


def semi_ce_loss(inputs, targets,
                 conf_mask=True, threshold=None,
                 threshold_neg=.0, temperature_value=1):
    # target => logit, input => logit
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets / temperature_value, dim=1)

        # for positive
        targets_real_prob = F.softmax(targets, dim=1)

        weight = targets_real_prob.max(1)[0]
        total_number = len(targets_prob.flatten(0))
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)

        mask_neg = (targets_prob < threshold_neg)

        neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype)
        if neg_label.shape[-1] != 21:
            neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
                                                           neg_label.shape[2], 21 - neg_label.shape[-1]]).cuda()),
                                  dim=3)
        neg_label = neg_label.permute(0, 3, 1, 2)
        neg_label = 1 - neg_label

        if not torch.any(mask):
            # neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
            # negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            # return zero, pass_rate, negative_loss_mat[mask_neg].mean_teacher()
            return zero, None, None
            
        else:
            positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight

            # neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
            # negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))

            # return positive_loss_mat[mask].mean_teacher(), pass_rate, negative_loss_mat[mask_neg].mean_teacher()
            return positive_loss_mat[mask].mean(), None, None
    else:
        raise NotImplementedError