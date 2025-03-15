class SVSoftmaxLoss(nn.Module):
    """
        Reference: <Large-Margin Softmax Loss using Synthetic Virtual Class>.
        Virtual Softmax: v = |W_yi|*(z_i / |z_i|)
        SV-Softmax: s = |W_yi|*(h / |h|), where h = m(z_i / |z_i|) - (1-m)(w_yi / |w_yi|)
        # Use v for misclassified, and s for correct ones
    """

    def __init__(self, in_features, num_classes, m=0.6):
        super(SVSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.m = m
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs, labels):
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
            w_yi_unit = F.normalize(W_yi, dim=0)
            z_i_norm = torch.norm(inputs, dim=1)
            z_i_unit = F.normalize(inputs, dim=1)

            # Virtual logit computation
            v_zi = W_yi_norm * z_i_norm
            v_zi = torch.clamp(v_zi, min=1e-10, max=50.0)
            # v_zi = torch.clamp(v_zi, min=1e-10, max=15.0)
            v_zi = v_zi.unsqueeze(1)

            # Synthetic logit computation
            h = self.m * z_i_unit - (1 - self.m) * w_yi_unit  # m [0,1]
            s = W_yi_norm.unsqueeze(1) * F.normalize(h, dim=1)
            s_zi = torch.einsum('ij,ij->i', s, inputs)
            s_zi = torch.clamp(s_zi, 1e-10, 50.0)
            # s_zi = torch.clamp(s_zi, 1e-10, 15.0)
            s_zi = s_zi.unsqueeze(1)

            # select and concat
            _, predicted_labels = WZ.max(dim=1)
            selected = torch.where(predicted_labels.unsqueeze(1) == labels.unsqueeze(1), s_zi, v_zi)
            WZ_new = torch.cat((WZ, selected), dim=1)
            return WZ_new
        else:
            return WZ
