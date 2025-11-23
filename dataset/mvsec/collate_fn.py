import torch

def collate_fn(batch):
    """
    batch: list of dataset samples
    returns a batched dictionary for GNN processing
    """

    # concatenate node features / attributes
    features = torch.cat([b["features"] for b in batch], dim=0)         # [total_N]
    positions = torch.cat([b["positions"] for b in batch], dim=0)       # [total_N,3]

    # build batch index for nodes
    batch_index = torch.cat([
        torch.full((b["positions"].shape[0],), i, dtype=torch.long)
        for i, b in enumerate(batch)
    ], dim=0)

    # merge edges (with offset)
    edges_list = []
    offset = 0
    for b in batch:
        e = b["edges"] + offset
        edges_list.append(e)
        offset += b["positions"].shape[0]
    edges = torch.cat(edges_list, dim=0)                                # [total_E,2]

    # stack flows (image-like tensor)
    flows = torch.stack([b["flow"] for b in batch], dim=0)              # [B,2,H,W]

    # concatenate events if needed for visualization or raw use
    raw_events = torch.cat([b["events"] for b in batch], dim=0)

    # metadata stored as simple python lists
    timestamps = [b["timestamp"] for b in batch]
    seq_names  = [b["sequence"] for b in batch]
    cameras    = [b["camera"] for b in batch]
    frame_ids  = [b["frame_idx"] for b in batch]

    return {
        "x": features,                   # node features
        "pos": positions,                # node coords (for GNN)
        "edge_index": edges,             # [E,2]
        "batch": batch_index,            # graph membership idx
        "flow": flows,                   # [B,2,H,W]
        "events": raw_events,            # optional raw tensor
        "timestamps": timestamps,
        "sequences": seq_names,
        "cameras": cameras,
        "frame_ids": frame_ids,
    }
