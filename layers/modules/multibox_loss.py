import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import ceil
from utils.box_utils import atss_match, ciou_loss, decode, log_sum_exp, match

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, cfg=None, use_atss=False, atss_topk=9, focal_alpha=0.25, focal_gamma=2.0, use_landmark=True):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.cfg = cfg
        self.use_atss = use_atss
        self.atss_topk = atss_topk
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_landmark = use_landmark
        self.num_priors_per_level = None
        if cfg is not None:
            # 根据当前输入尺寸和每层 anchor 设置，预先计算各层 prior 数量，供 ATSS 分层采样使用。
            self.num_priors_per_level = self._build_num_priors_per_level(cfg)

    def _build_num_priors_per_level(self, cfg):
        image_size = cfg['image_size']
        priors_per_level = []
        for step, min_sizes in zip(cfg['steps'], cfg['min_sizes']):
            feature_h = ceil(image_size / step)
            feature_w = ceil(image_size / step)
            priors_per_level.append(feature_h * feature_w * len(min_sizes))
        return priors_per_level

    def focal_loss(self, conf_data, conf_t):
        # Focal Loss：降低大量易分类负样本对分类分支的主导作用。
        conf_flat = conf_data.view(-1, self.num_classes)
        target_flat = conf_t.view(-1)
        ce_loss = F.cross_entropy(conf_flat, target_flat, reduction='none')
        prob = F.softmax(conf_flat, dim=1)
        pt = prob[torch.arange(prob.size(0), device=prob.device), target_flat]
        alpha_t = torch.where(target_flat > 0, torch.full_like(pt, self.focal_alpha), torch.full_like(pt, 1 - self.focal_alpha))
        loss = alpha_t * torch.pow(1 - pt, self.focal_gamma) * ce_loss
        return loss.sum()

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        device = loc_data.device
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.zeros(num, num_priors, 4, device=device)
        landm_t = torch.zeros(num, num_priors, 10, device=device)
        bbox_t = torch.zeros(num, num_priors, 4, device=device)
        conf_t = torch.zeros(num, num_priors, dtype=torch.long, device=device)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            if self.use_atss and self.num_priors_per_level is not None:
                # 使用 ATSS 替换原始 IoU 阈值匹配。
                atss_match(defaults, truths, labels, landms, self.variance, self.num_priors_per_level, loc_t, conf_t, landm_t, bbox_t, idx, topk=self.atss_topk)
            else:
                match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, bbox_t, idx)

        zeros = torch.tensor(0, device=device)
        if self.use_landmark:
            # 保留原始 landmark 分支训练逻辑，适用于带五点标注的数据集。
            pos1 = conf_t > zeros
            num_pos_landm = pos1.long().sum(1, keepdim=True)
            N1 = max(num_pos_landm.data.sum().float(), 1)
            pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
            landm_p = landm_data[pos_idx1].view(-1, 10)
            landm_t = landm_t[pos_idx1].view(-1, 10)
            loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        else:
            # 自建 bbox-only 数据集没有关键点监督时，显式关闭 landmark loss。
            N1 = 1
            loss_landm = landm_data.sum() * 0


        pos = conf_t != zeros
        num_pos = pos.long().sum(1, keepdim=True)

        if self.use_atss:
            conf_target = conf_t.clone()
            # RetinaFace 当前是单类别检测，正样本统一映射到前景类别 1。
            conf_target[pos] = 1

            # 使用 ATSS 时，回归分支切换为 CIoU Loss。
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            matched_priors = priors.unsqueeze(0).expand(num, num_priors, 4)[pos_idx].view(-1, 4)
            matched_boxes = bbox_t[pos_idx].view(-1, 4)
            if loc_p.numel() == 0:
                loss_l = loc_data.sum() * 0
            else:
                # 先将预测偏移量解码回真实框，再与 GT 框计算 CIoU 损失。
                decoded_boxes = decode(loc_p, matched_priors, self.variance)
                loss_l = ciou_loss(decoded_boxes, matched_boxes, reduction='sum')

            # 使用 ATSS 时，分类分支同步切换为 Focal Loss。
            loss_c = self.focal_loss(conf_data, conf_target)
        else:
            # 关闭 ATSS 时，恢复原始 RetinaFace 的 SmoothL1 边框回归损失。
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t_pos = loc_t[pos_idx].view(-1, 4)
            if loc_p.numel() == 0:
                loss_l = loc_data.sum() * 0
            else:
                loss_l = F.smooth_l1_loss(loc_p, loc_t_pos, reduction='sum')

            # 关闭 ATSS 时，恢复原始的 CrossEntropy + Hard Negative Mining。
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

            loss_c = loss_c.view(num, -1)
            loss_c[pos] = 0
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos_for_neg = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos_for_neg, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            if conf_p.numel() == 0:
                loss_c = conf_data.sum() * 0
            else:
                loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
