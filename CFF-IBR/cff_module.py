import torch
import torch.nn as nn
import torch.nn.functional as F


class CFFModule(nn.Module):
    def __init__(self, resolutions=(32, 64), feat_dim=8, out_dim=16,
                 bbox_size=4.0, auto_bbox_warmup=100):
        super(CFFModule, self).__init__()

        self.resolutions = list(resolutions)
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.auto_bbox_warmup = auto_bbox_warmup

        # ---------- step counter for diagnostic printing ----------
        self._step_count = 0

        # ---------- auto-bbox buffers (saved in checkpoint) ----------
        self.register_buffer('center', torch.zeros(3))
        self.register_buffer('scale', torch.tensor(float(bbox_size)))
        self.register_buffer('bbox_ready', torch.tensor(0, dtype=torch.long))

        # ---------- multi-resolution feature grids ----------
        self.grids = nn.ParameterList()
        for res in self.resolutions:
            grid = nn.Parameter(torch.randn(1, feat_dim, res, res, res) * 0.01)
            self.grids.append(grid)

        # ---------- projection to target dimension ----------
        total_feat = feat_dim * len(self.resolutions)
        self.proj = nn.Sequential(
            nn.Linear(total_feat, out_dim),
            nn.ELU(inplace=True)
        )
        self._init_proj_weights()

    # ------------------------------------------------------------------
    def _init_proj_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _accumulate_bbox_stats(self, pts):
        """Collect point cloud statistics during warmup to auto-compute bbox.
        Uses 5%-95% percentile to filter outliers for stable bbox estimation.
        Subsamples each batch to limit memory usage."""
        pts_flat = pts.reshape(-1, 3)

        # ---- each batch retains up to 5000 points to avoid bursting memory during quantile  ----
        max_pts_per_batch = 5000
        if pts_flat.shape[0] > max_pts_per_batch:
            idx = torch.randperm(pts_flat.shape[0])[:max_pts_per_batch]
            pts_flat = pts_flat[idx]

        if not hasattr(self, '_all_pts') or self._all_pts is None:
            self._all_pts = [pts_flat.cpu()]
            self._warmup_count = 0
        else:
            self._all_pts.append(pts_flat.cpu())

        self._warmup_count += 1

        if self._warmup_count >= self.auto_bbox_warmup:
            all_pts = torch.cat(self._all_pts, dim=0)  # [N_total, 3] ≤ 500k

            # if still too large, sample again
            max_total = 500000
            if all_pts.shape[0] > max_total:
                idx = torch.randperm(all_pts.shape[0])[:max_total]
                all_pts = all_pts[idx]

            # use 5% and 95% percentile to filter outliers
            pts_min = torch.quantile(all_pts, 0.05, dim=0)
            pts_max = torch.quantile(all_pts, 0.95, dim=0)

            center = (pts_min + pts_max) / 2.0
            half_range = (pts_max - pts_min) / 2.0
            max_half = half_range.max().item() * 1.1  # 10% margin

            self.center.copy_(center.to(self.center.device))
            self.scale.fill_(max_half)
            self.bbox_ready.fill_(1)

            print(f"\n{'=' * 55}")
            print(f"[CFF] Auto-bbox computed from {self._warmup_count} batches "
                  f"({all_pts.shape[0]} points used):")
            print(f"  Percentile 5%   : [{pts_min[0]:.2f}, {pts_min[1]:.2f}, {pts_min[2]:.2f}]")
            print(f"  Percentile 95%  : [{pts_max[0]:.2f}, {pts_max[1]:.2f}, {pts_max[2]:.2f}]")
            print(f"  Scene center    : [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
            print(f"  Scale (half)    : {max_half:.2f}")
            for res in self.resolutions:
                voxel = 2.0 * max_half / res
                print(f"  {res:>3d}^3 voxel size: {voxel:.4f} world units")
            print(f"{'=' * 55}\n")

            # cleanup
            self._all_pts = None

    # ------------------------------------------------------------------
    def forward(self, pts):
        if self.bbox_ready.item() == 0:
            self._accumulate_bbox_stats(pts)

        pts_norm = (pts - self.center) / self.scale
        N_rays, N_samples, _ = pts.shape
        grid_pts = pts_norm.view(1, N_rays, N_samples, 1, 3)

        feats = []
        for grid in self.grids:
            sampled = F.grid_sample(grid, grid_pts,
                                    align_corners=True,
                                    mode='bilinear',
                                    padding_mode='zeros')
            sampled = sampled.squeeze(0).squeeze(-1)
            sampled = sampled.permute(1, 2, 0)
            feats.append(sampled)

        feats = torch.cat(feats, dim=-1)
        out = self.proj(feats)

        if self.training:
            self._step_count += 1
            if self._step_count % 2000 == 0:
                print(f"[CFF-no-gate] step={self._step_count} "
                      f"cff_mag={out.abs().mean().item():.6f}")

        return out

    # ------------------------------------------------------------------
    def compute_tv_loss(self):
        tv = 0.0
        for grid in self.grids:
            tv = tv + (grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]).pow(2).mean()
            tv = tv + (grid[:, :, :, 1:, :] - grid[:, :, :, :-1, :]).pow(2).mean()
            tv = tv + (grid[:, :, :, :, 1:] - grid[:, :, :, :, :-1]).pow(2).mean()
        return tv