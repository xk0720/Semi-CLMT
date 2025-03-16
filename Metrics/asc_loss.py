import torch
import torch.nn as nn
# from nets.project_head import ProjHead


class ASC_loss(nn.Module):
    def __init__(self, batch_size, device, sur_siml='dice', pHead_sur='set_false'):
        super(ASC_loss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)  # 用来提取 negative samples 的索引
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sur_siml = sur_siml
        self.pHead_sur = pHead_sur
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.similar_dice = BinaryDice_xent()

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)  # (N, N)
        mask = mask.fill_diagonal_(0)  # 对角线为False, 其余为True
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):  # zi, zj (B, Class, H, W)
        N = 2 * self.batch_size
        # if self.pHead_sur == 'set_true':
        #     projHead_sur = ProjHead().cuda()
        #     z_i_head = projHead_sur(z_i)
        #     z_j_head = projHead_sur(z_j)
        #     z_head = torch.cat((z_i_head, z_j_head), dim=0)

        z = torch.cat((z_i, z_j), dim=0)  # z (N(2B), Class, H, W)
        if self.sur_siml == 'cos' and self.pHead_sur == 'set_false':
            z_flatten = torch.flatten(z, start_dim=1)
            sim_sur = self.similarity_f(z_flatten.unsqueeze(1), z_flatten.unsqueeze(0))
        elif self.sur_siml == 'dice' and self.pHead_sur == 'set_false':
            # z.unsqueeze(1) (N, 1, Class, H, W)  z.unsqueeze(0) (1, N, Class, H, W)
            sim_sur = self.similar_dice(z.unsqueeze(1), z.unsqueeze(0))  # (N, N)
        # elif self.pHead_sur == 'set_true':
        #     sim_sur = self.similarity_f(z_head.unsqueeze(1), z_head.unsqueeze(0))

        sim = sim_sur

        # contrastive learning

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # positive_samples (N,)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)  # similar to infoNCE loss
        loss /= N
        return loss


class BinaryDice_xent(nn.Module):
    def __init__(self):
        super(BinaryDice_xent, self).__init__()

    def _dice(self, score, target):  # score (N, 1, Class, H, W)  target (1, N, Class, H, W)
        smooth = 1e-6
        dim_len = len(score.size())

        #-------------------------------------------------------------------------------------
        
        """
        Option_1: Dice score
        """

        # if dim_len == 5:
        #     dim = (2, 3, 4)
        # elif dim_len == 4:
        #     dim = (2, 3)
        #
        # intersect = torch.sum(score * target, dim=dim)  # intersect (N, N)
        # y_sum = torch.sum(target * target, dim=dim)  # y_sum (1, N)
        # z_sum = torch.sum(score * score, dim=dim)  # z_sum (N, 1)
        # dice_sim = (2 * intersect + smooth) / (z_sum + y_sum + smooth)  # (N, N)

        """
        Option_2: mIoU score
        """
        # reducing only spatial dimensions (not batch nor channels)
        if dim_len == 5:
            dim = (3, 4)
        elif dim_len == 4:
            dim = (3,)

        intersect = torch.sum(score * target, dim=dim)  # intersect (N, N, Class)
        target_o = torch.sum(target, dim=dim) # target_o (1, N, Class)
        score_o = torch.sum(score, dim=dim) # score_o (N, 1, Class)
        union = target_o + score_o - intersect # (N, N, Class)

        similarity = torch.where(union > 0, (intersect) / union, torch.tensor(1.0, device=target_o.device))
        t_zero = torch.zeros(1, device=similarity.device, dtype=similarity.dtype)
        nans = torch.isnan(similarity)
        not_nans = (~nans).float()
        not_nans = not_nans.sum(dim=-1) # channel average, (dimension -1)

        dice_sim = torch.where(not_nans > 0, similarity.sum(dim=-1) / not_nans, t_zero) # (N, N)
        
        #-------------------------------------------------------------------------------------

        
        return dice_sim # return similarity

    def forward(self, inputs, target):  # input (N, 1, Class, H, W)  target (1, N, Class, H, W)
        assert inputs.size()[2:] == target.size()[2:], 'predict & target shape do not match'
        dice_sim = self._dice(inputs, target)  # (N, N)
        return dice_sim
