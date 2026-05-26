import torch
from packaging import version
from torch import nn


class PatchNCELoss(nn.Module):
    """Implements PatchNCELoss."""

    def __init__(self, nce_t=0.07, gradient_flows_to_negative_nce=False, nce_fake_negatives=True):
        super().__init__()
        self.nce_t = nce_t  # temparature
        self.gradient_flows_to_negative_nce = gradient_flows_to_negative_nce
        self.nce_fake_negatives = nce_fake_negatives
        self.index = 0
        self.loss = nn.LogSoftmax(1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse("1.2.0") else torch.bool

    def forward(self, feat_q, feat_k):
        """
        feat_q: recons_feats
        feat_k: target_feats
        """
        # import pdb
        # pdb.set_trace()
        batch_size_real, num_patches, dim = feat_q.size()
        batch_size = batch_size_real * num_patches
        feat_q = feat_q.view(-1, dim)  # [batch_size * num_patches, dim]
        feat_k = feat_k.view(-1, dim)

        if not self.gradient_flows_to_negative_nce:
            feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(batch_size, 1, -1),
            feat_k.view(batch_size, -1, 1),
        )  # num_patches*bs, 1, 1
        l_pos = l_pos.view(batch_size, 1)  # [bs * num_patches, 1]

        # negative logits
        if self.nce_fake_negatives:
            l_neg = torch.empty((2 * batch_size, 0), device=feat_q.device)
        else:
            l_neg = torch.empty((batch_size, 0), device=feat_q.device)

        # neg logit -- current batch
        # reshape features to batch size
        feat_q = feat_q.view(batch_size_real, -1, dim)
        feat_k = feat_k.view(batch_size_real, -1, dim)
        npatches = feat_q.size(1)

        if self.nce_fake_negatives:  # including all patch considered as negative of fake sample
            feat_k = torch.cat((feat_k, feat_q), dim=1)

        l_neg_curbatch = torch.bmm(
            feat_q, feat_k.transpose(2, 1)
        )  # shape B,num_patches, num_patches*2, each row in fake position coresspond to all other spatial position in both real and fake

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(
            npatches,
            device=feat_q.device,
            dtype=self.mask_dtype,
        )[None, :, :]

        if self.nce_fake_negatives:
            diagonal = torch.cat(
                (diagonal, diagonal), dim=-1
            )  # shape 1, num_patches, num_patches*2 , mask out all value in diagonal, which is the similarity between feature at the same position of both real and fake

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg_curbatch = l_neg_curbatch.view(-1, npatches)

        # assign l_neg_curpatch to l_neg
        l_neg = torch.cat((l_neg, l_neg_curbatch), dim=1)

        if self.nce_fake_negatives:
            l_pos = l_pos.view(batch_size_real, npatches, 1)
            l_neg = l_neg.view(batch_size_real, npatches, -1)
            out = torch.cat((l_pos, l_neg), dim=2)
            out = out.view(batch_size_real * num_patches, -1)
        else:
            out = torch.cat((l_pos, l_neg), dim=1)

        out = torch.div(out, self.nce_t)
        return self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )  # label is a batch of zeros indicate the first index ( 0 index) as label of current sample (row) which is the positive case
