import torch

def AEE(pred, gt):
    mag = torch.linalg.norm(gt, dim=1)
    valid_mask = mag > 1e-6

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    diff = torch.linalg.norm(pred[valid_mask] - gt[valid_mask], dim=1)
    return diff.mean()


def percent_outliers(pred, gt):
    mag = torch.linalg.norm(gt, dim=1)
    valid_mask = mag > 1e-6

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred = pred[valid_mask]
    gt   = gt[valid_mask]

    epe = torch.linalg.norm(pred - gt, dim=1)
    gt_mag = torch.linalg.norm(gt, dim=1)

    threshold = torch.clamp(0.05 * gt_mag, min=3.0)
    return (epe > threshold).float().mean() * 100.0


def flow_accuracy(pred, gt, zeta=0.25):
    mag = torch.linalg.norm(gt, dim=1)
    valid_mask = mag > 1e-6

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred = pred[valid_mask]
    gt   = gt[valid_mask]

    rel = torch.linalg.norm(pred - gt, dim=1) / (torch.linalg.norm(pred, dim=1) + 1e-12)
    return (rel < zeta).float().mean()

