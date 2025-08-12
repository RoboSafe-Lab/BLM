import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
def transform_points_torch(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
    def _transform(points, transf_matrix):
        num_dims = transf_matrix.shape[-1] - 1
        rotation = transf_matrix[..., :num_dims, :num_dims]
        translation = transf_matrix[..., :num_dims, -1]
        return torch.matmul(points, rotation.transpose(-1, -2)) + translation.unsqueeze(-2)

    if points.ndim == 4 and transf_matrix.ndim == 3:
        B, N, T, F = points.shape
        points_reshaped = points.view(B, N * T, F)  # (B, N*T, F)
        transformed = _transform(points_reshaped, transf_matrix)  # (B, N*T, F)
        return transformed.view(B, N, T, F)

    elif points.ndim == 3 and transf_matrix.ndim == 3:
        return _transform(points, transf_matrix)

    elif points.ndim == 2 and transf_matrix.ndim == 2:
        return _transform(points.unsqueeze(0), transf_matrix.unsqueeze(0))[0]

    elif points.ndim == 3 and transf_matrix.ndim == 2:
        return _transform(points, transf_matrix.unsqueeze(0))

    else:
        raise NotImplementedError(
            f"unsupported case: points shape {points.shape}, matrix shape {transf_matrix.shape}"
        )

def plot_trajdata_batch(batch,preds_positions):
    fix,axes = plt.subplots(1,2,figsize=(20,10))
    
    for i, idx in enumerate([0,-1]):
        image = batch["maps"][idx].cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        map_size = image.shape[1]
        rgb_map = np.zeros((map_size, map_size, 3))
        rgb_map[image[:, :, 0] > 0] = np.array([129, 195, 228]) / 255.
        rgb_map[image[:, :, 1] > 0] = np.array([164, 184, 196]) / 255.
        rgb_map[image[:, :, 2] > 0] = np.array([96, 117, 138]) / 255.
        rgb_map[rgb_map.sum(axis=-1) == 0] = [0.95, 0.95, 0.95]
        axes[i].imshow(rgb_map, extent=[0, map_size, map_size, 0])

        # drivable_map = batch["drivable_map"][idx].cpu()
        # axes[i].imshow(drivable_map, extent=[0, map_size, map_size, 0])
        raster_from_agent = batch["raster_from_center"][idx].cpu()
        # 先绘制邻居位置点（最底层，数量最多）
        neighbor_pos = batch["neigh_hist_positions"][idx].cpu()
        neighbor_raster = transform_points_torch(neighbor_pos, raster_from_agent)
        axes[i].scatter(neighbor_raster[..., 0], neighbor_raster[..., 1], c='green', s=3, alpha=0.6, label='Neighbors History')

        # 然后绘制历史位置点（中间层）
        his_pos = batch["center_hist_positions"][idx].cpu()
        his_raster = transform_points_torch(his_pos, raster_from_agent)
        axes[i].scatter(his_raster[..., 0], his_raster[..., 1], c='blue', s=8, alpha=0.8, label='Ego History')

        # 最后绘制未来位置点（顶层，最重要）
        fut_pos = batch["center_fut_positions"][idx].cpu()
        
        fut_raster = transform_points_torch(fut_pos, raster_from_agent)
        axes[i].scatter(fut_raster[..., 0], fut_raster[..., 1], c='red', s=12, alpha=0.9, label='Ego Future')

        # preds_raster = transform_points_torch(preds_positions[idx].cpu(), raster_from_agent)
        # axes[i].scatter(preds_raster[..., 0], preds_raster[..., 1], c='yellow', s=5, alpha=0.9, label='Ego Predictions')

        axes[i].set_title(f'Sample {idx}')
        # 只在第一个子图上添加图例
  
    
    plt.show()
    