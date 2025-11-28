import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_edge(alpha, row, num_nodes=None, eps=1e-12):
    """
    alpha: [E, F] attention logits
    row:   [E] target index (grouping key)
    returns per-edge softmax over edges that share the same row
    """
    if num_nodes is None:
        num_nodes = int(row.max()) + 1

    # 1) compute max per node
    max_per_node = torch.full((num_nodes, alpha.size(1)),
                              -1e9, device=alpha.device, dtype=alpha.dtype)
    max_per_node.index_put_((row,), alpha, accumulate=True)
    # Actually accumulate loses track, so compute real max via scatter:
    max_per_node = torch.zeros_like(max_per_node).scatter_reduce_(0, row[:,None].expand_as(alpha),
                                                                  alpha, reduce='amax')

    # 2) stabilize alpha
    alpha_stable = alpha - max_per_node[row]

    # 3) exponentiate
    exp_alpha = torch.exp(alpha_stable)

    # 4) sum per node
    sum_per_node = torch.zeros((num_nodes, alpha.size(1)),
                               device=alpha.device, dtype=alpha.dtype)
    sum_per_node.scatter_add_(0, row[:,None].expand_as(alpha), exp_alpha)

    # 5) normalize
    softmax_vals = exp_alpha / (sum_per_node[row] + eps)
    return softmax_vals


class MyPointTransformerConv(nn.Module):
    """
    Simplified PointTransformerConv-style layer:
    - No quantization
    - No calibration modes
    - Uses attention based on feature & positional differences
    - Aggregation = SUM
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Drop-in replacements for PointTransformer structure
        self.lin = nn.Linear(input_dim, output_dim, bias=bias)       # main feature transform (W3)
        self.lin_src = nn.Linear(input_dim, output_dim, bias=False)  # W1
        self.lin_dst = nn.Linear(input_dim, output_dim, bias=False)  # W2

        self.pos_nn = nn.Linear(3, output_dim)                       # h_Theta positional MLP

        self.attn_nn = nn.Sequential(                                # γ_Theta attention MLP
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

        self.use_relu = True

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.pos_nn.reset_parameters()
        for m in self.attn_nn:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,          # [N, Fin]
        pos: torch.Tensor,        # [N, 3]
        edge_index: torch.Tensor  # [E, 2] (i <- j)
    ) -> torch.Tensor:
        
        row = edge_index[:, 0]  # target nodes (i)
        col = edge_index[:, 1]  # source nodes (j)

        # feature projections
        x_proj = self.lin(x)            # [N, Fout]
        alpha_src = self.lin_src(x)     # W1 * x_i
        alpha_dst = self.lin_dst(x)     # W2 * x_j

        x_j = x_proj[col]               # features for edges
        alpha_i = alpha_src[row]
        alpha_j = alpha_dst[col]

        pos_i = pos[row]
        pos_j = pos[col]

        # positional encoding delta_ij
        delta = self.pos_nn(pos_i - pos_j)

        # compute attention
        alpha = alpha_i - alpha_j + delta
        alpha = self.attn_nn(alpha)
        att = softmax_edge(alpha, row)

        # weighted message
        msg = att * (x_j + delta)

        # SUM aggregation
        out = torch.zeros(x.size(0), self.output_dim, device=x.device)
        out.index_add_(0, row, msg)

        return out
