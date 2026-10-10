from __future__ import annotations
import math
import numpy as np
import torch
def compute_advantages(trajectory_groups_P: List[TrajectoryGroup], adv_estimator: str, adv_estimator_beta: float, adv_estimator_mu: float = 5/1.503163635, adv_estimator_sigma: float = 0.000001) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        if adv_estimator == "mean_baseline":
            advantages_G = rewards_G - rewards_G.mean()
        elif adv_estimator == "entropic":
            beta = adv_estimator_beta
            s_safe = rewards_G - rewards_G.max(dim=-1, keepdim=True)[0]
            e = torch.exp(beta * s_safe)
            k = e.shape[0]
            if k == 1:
                Z = e 
            else:
                Z = (e.sum() - e) / (k - 1)
            w = e / (Z + 1e-12)
            advantages_G = w - 1.0
        elif adv_estimator == "entropic_adaptive_beta":
            delta = np.log(2)
            beta_max = 1e6
            iters = 60
            eps = 1e-12

            r = rewards_G.float()
            k = r.shape[0]

            if k < 2:
                beta = r.new_tensor(0.0)
            else:
                logK = math.log(k)

                def kl_hat(beta_scalar: float) -> float:
                    # q_beta over samples: q ∝ exp(beta * r), KL(q||uniform)
                    b = r.new_tensor(beta_scalar)
                    logits = b * (r - r.max(dim=0, keepdim=True).values)      # stable
                    logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                    q = torch.exp(logq)
                    kl = (q * (logq + logK)).sum(dim=0)   
                    return float(kl.mean().item())    

                lo, hi = 0.0, 1.0
                if kl_hat(hi) < delta:
                    while hi < beta_max and kl_hat(hi) < delta:
                        hi *= 2.0
                    if kl_hat(hi) < delta:
                        beta = r.new_tensor(hi)  # best effort
                    else:
                        beta = None
                else:
                    beta = None

                if beta is None:
                    for _ in range(iters):
                        mid = 0.5 * (lo + hi)
                        if kl_hat(mid) < delta:
                            lo = mid
                        else:
                            hi = mid
                    beta = r.new_tensor(hi)

            # LOO entropic advantages using solved beta
            e = torch.exp(beta * (r - r.max(dim=0, keepdim=True).values))

            if k == 1:
                Z = e
            else:
                Z = (e.sum(dim=0, keepdim=True) - e) / (k - 1)

            w = e / (Z + eps)
            advantages_G = w - 1.0
        else:
            raise ValueError(f"Invalid advantage estimator: {adv_estimator}")
        advantages_P.append(advantages_G)

    return advantages_P

