import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets)


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.1, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.margin)
        cosine.acos_()
        cosine[index] += m_hot

        cosine.cos_().mul_(self.scale)
        max_cosine = torch.max(cosine, dim=1, keepdim=True)[0]
        return F.cross_entropy(cosine-max_cosine, label)


class CosFaceLoss(nn.Module):
    def __init__(self, margin=0.35, scale=64):
        super(CosFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        one_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device)
        one_hot.scatter_(1, labels[index, None], 1)
        logits = logits - one_hot * self.margin
        logits = logits * self.scale

        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        return F.cross_entropy(logits - max_logits, labels)


class LMSoftmaxLoss(nn.Module):
    """
        Zhou X, Liu X, Zhai D, et al. LEARNING TOWARDS THE LARGEST MARGINS[C]
        //10th International Conference on Learning Representations, ICLR 2022. 2022.
    """
    def __init__(self, scale=64, weight=None):
        super(LMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.weight = weight
        self.eps = 1e-7

    def forward(self, logits, labels):
        mask = F.one_hot(labels, logits.size()[1]).float().to(logits.device)
        logits = logits * self.scale
        l1 = torch.sum(logits * mask, dim=1, keepdim=True)
        diff = logits - l1
        logits_exp = torch.exp(diff)
        l2 = torch.sum(logits_exp * (1 - mask), dim=1)
        loss = torch.log(l2)
        if self.weight is not None:
            weight = self.weight.gather(0, labels.view(-1))
            loss = loss * weight
        return loss.mean() / self.scale


class SampleMarginLoss(nn.Module):
    """
        Zhou X, Liu X, Zhai D, et al. LEARNING TOWARDS THE LARGEST MARGINS[C]
        //10th International Conference on Learning Representations, ICLR 2022. 2022.
    """
    def __init__(self):
        super(SampleMarginLoss, self).__init__()

    def forward(self, logits, labels):
        label_one_hot = F.one_hot(labels, logits.size()[1]).float().to(logits.device)
        l1 = torch.sum(logits * label_one_hot, dim=1)
        tmp = logits * (1 - label_one_hot) - label_one_hot
        l2 = torch.max(tmp, dim=1)[0]
        loss = l2 - l1
        return loss.mean()


class NormFaceLoss(nn.Module):
    def __init__(self, scale=64, weight=None):
        super(NormFaceLoss, self).__init__()
        self.scale = scale
        self.weight = weight

    def forward(self, logits, labels):
        logits = logits * self.scale
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        return F.cross_entropy(logits - max_logits, labels, weight=self.weight)


class SphereFaceLoss(nn.Module):
    def __init__(self, scale=30., margin=1.5):
        super(SphereFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, weight, x, y):
        weight = weight.T
        with torch.no_grad():
            weight.data = F.normalize(weight.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(x, dim=1).mm(weight)
        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1. + 1e-5, 1. - 1e-5))
            m_theta.scatter_(
                1, y.view(-1, 1), self.margin, reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - cos_theta

        logits = self.scale * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)

        return loss


class Unified_Cross_Entropy_Loss(nn.Module):
    """
        Zhou J, Jia X, Li Q, et al. Uniface: Unified cross-entropy loss for deep face recognition[C]
        //Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 20730-20739.
    """
    def __init__(self, in_features, out_features, m=0.4, s=64):
        super(Unified_Cross_Entropy_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.bias = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features * 10))
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input, label, mode):
        if mode:
            cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))

            cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
            cos_m_theta_n = self.s * cos_theta - self.bias
            p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
            n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s)))

            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool, device=cos_theta.device)
            # one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            loss = one_hot * p_loss + (~one_hot) * n_loss
            return loss.sum(dim=1).mean()
        else:
            return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', m=' + str(self.m) \
            + ', s=' + str(self.s) + ')'


class WSoftmax(nn.Module):
    """
        Li X, Wang W. Learning discriminative features via weights-biased softmax loss[J].
        Pattern Recognition, 2020, 107: 107405.
    """
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, W, x, labels):
        """
        Args:
            x: input features (N, D)
            W: classifier weights (C, D)
            labels: ground truth labels (N,)
        Returns:
            logits: modified logits (N, C)
        """
        W_norm = F.normalize(W, p=2, dim=1)  # (C, D)

        original_logits = x @ W_norm.T  # (N, C)
        W_c = W_norm[labels]
        combined_W = self.alpha * W_c.unsqueeze(1) + W_norm.unsqueeze(0)  # (N, C, D)
        combined_W_norm = F.normalize(combined_W, p=2, dim=2)  # (N, C, D)

        adjusted_logits = torch.bmm(
            combined_W_norm,
            x.unsqueeze(2)  # (N, D, 1)
        ).squeeze(2)  # (N, C)
        mask = torch.zeros_like(adjusted_logits, dtype=torch.bool)
        mask.scatter_(1, labels.unsqueeze(1), True)
        final_logits = torch.where(mask, original_logits, adjusted_logits)
        loss = F.cross_entropy(final_logits, labels)
        return loss


class SO_Softmax(nn.Module):
    """
        Zhang Q, Yang J, Zhang X, et al. SO-softmax loss for discriminable embedding
        learning in CNNs[J]. Pattern Recognition, 2022, 131: 108877.
    """
    def __init__(self, lambda_p: float = 2.0, lambda_n: float = 2.0, m: float = 0.3):
        """
        Args:
            lambda_p (float): The scaling factor for positive examples.
            lambda_n (float): The scaling factor for negative examples.
            m (float): The relaxation factor.
        """
        super(SO_Softmax, self).__init__()
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): A matrix of normalized cosine similarities.
                                   Shape: [batch_size, num_classes].
            labels (torch.Tensor): Ground truth labels.
                                   Shape: [batch_size,].
                                   (Note: The original comment mentioned ignoring samples with label -1,
                                   but this implementation does not handle that. See notes below.)
        """
        batch_size, num_classes = logits.shape

        s_i = logits[torch.arange(batch_size), labels]  # (batch_size,)
        pos_term = self.lambda_p * (self.m ** 2 - (s_i - 1) ** 2)
        positive_term = torch.exp(pos_term)  # (batch_size,)

        neg_terms = -self.lambda_n * (self.m ** 2 - logits ** 2)
        negative_terms = torch.exp(neg_terms)  # (batch_size, num_classes)

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = True

        negative_terms = negative_terms.masked_fill(mask, 0.0)
        negative_sum = negative_terms.sum(dim=1)  # (batch_size,)

        denominator = negative_sum + positive_term
        losses = -torch.log(positive_term / denominator)
        return losses.mean()


class VirtualSoftmax(nn.Module):
    """
        Chen B, Deng W, Shen H. Virtual class enhanced discriminative embedding learning[J].
         Advances in Neural Information Processing Systems, 2018, 31.
    """
    def __init__(self, in_features, num_classes, scale=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs, labels):
        weight = self.weight.T
        WX = torch.matmul(inputs, weight)
        if self.training:
            W_yi = weight[:, labels]
            W_yi_norm = torch.norm(W_yi, dim=0)
            X_i_norm = torch.norm(inputs, dim=1)
            WX_virt = W_yi_norm * X_i_norm * self.scale
            WX_virt = torch.clamp(WX_virt, min=1e-10, max=15.0)
            WX_virt = WX_virt.unsqueeze(1)
            WX_new = torch.cat([WX, WX_virt], dim=1)
            return WX_new
        else:
            return WX


class SVSoftmaxLoss(nn.Module):
    """
        Reference: <Large-Margin Softmax Loss using Synthetic Virtual Class>.
        Virtual Softmax: v = |W_yi|*(z_i / |z_i|)
        SV-Softmax: s = |W_yi|*(h / |h|), where h = m(z_i / |z_i|) - (1-m)(w_yi / |w_yi|)
        # Use v for misclassified, and s for correct ones
    """

    def __init__(self, in_features, num_classes, m=0.6):
        super().__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.m = m
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs, labels, **kwargs):
        """
            Args:
                - v_zi: v*z_i, virtual class logits
                - s_zi: s*z_i, synthetic class logits
        """
        weight_t = self.weight.T
        WZ = torch.matmul(inputs, weight_t)
        if self.training:
            W_yi = weight_t[:, labels]
            W_yi_norm = torch.norm(W_yi, dim=0)
            w_yi_unit = F.normalize(W_yi.t(), dim=1)
            z_i_norm = torch.norm(inputs, dim=1)
            z_i_unit = F.normalize(inputs, dim=1)

            # Virtual logit computation
            v_zi = W_yi_norm * z_i_norm
            v_zi = torch.clamp(v_zi, min=1e-10, max=50.0)   # for mnist and cifar10
            # v_zi = torch.clamp(v_zi, min=1e-10, max=15.0) # for all the other datasets mentioned in the paper
            v_zi = v_zi.unsqueeze(1)

            # Synthetic logit computation
            h = self.m * z_i_unit - (1 - self.m) * w_yi_unit  # m [0,1]
            s = W_yi_norm.unsqueeze(1) * F.normalize(h, dim=1)
            s_zi = torch.einsum('ij,ij->i', s, inputs)
            s_zi = torch.clamp(s_zi, 1e-10, 50.0)   # for mnist and cifar10
            # s_zi = torch.clamp(s_zi, 1e-10, 15.0) # for all the other datasets mentioned in the paper
            s_zi = s_zi.unsqueeze(1)

            # select and concat
            _, predicted_labels = WZ.max(dim=1)
            selected = torch.where(predicted_labels.unsqueeze(1) == labels.unsqueeze(1), s_zi, v_zi)
            WZ_new = torch.cat((WZ, selected), dim=1)
            return WZ_new
        else:
            return WZ
