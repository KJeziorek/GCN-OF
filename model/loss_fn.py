import torch
import torch.nn.functional as F

def sample_flow_at_nodes(flow, pos, batch_idx):
    """
    flow: [B,2,H,W]
    pos:  [N,3]  (x,y,t_norm), but only x,y used
    batch_idx: [N]
    returns [N,2]
    """
    B, C, H, W = flow.shape
    x = pos[:,0].long().clamp(0, W-1)
    y = pos[:,1].long().clamp(0, H-1)

    return flow[batch_idx, :, y, x].permute(0,1)   # [N,2]

def smooth_l1_loss(pred, gt, beta=0.025):
    diff = pred - gt
    abs_diff = diff.abs()
    loss = torch.where(
        abs_diff < beta,
        0.5 * (diff * diff) / beta,
        abs_diff - 0.5 * beta
    )
    return loss.mean()

def graph_charbonnier_loss(pred, edge_index, alpha=0.5, eps=1e-3):
    # pred: [N,2]
    # edge_index: [E,2]

    src = edge_index[:,0]
    dst = edge_index[:,1]

    diff = pred[src] - pred[dst]
    diff_norm_sq = (diff * diff).sum(dim=1)

    deg = torch.bincount(src, minlength=pred.size(0)).float()
    weight = 1.0 / (deg[src] + 1e-6)

    return torch.mean(weight * (diff_norm_sq + eps**2)**alpha)

def optical_flow_loss(pred, gt, edge_index):
    l_sl1 = smooth_l1_loss(pred, gt)
    l_smooth = graph_charbonnier_loss(pred, edge_index)

    return l_sl1 + 0.1 * l_smooth, l_sl1, l_smooth
