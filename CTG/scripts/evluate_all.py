import itertools
import json

import pandas as pd
import h5py
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

# from scripts.reconstruction_helper import ReconstructionHelper


def calculate_jsd_kde(gt_data: np.ndarray, sim_data: np.ndarray, n_points: int = 1000,
                      const_tol: float = 1e-4) -> float:
    gt_std, sim_std = np.std(gt_data), np.std(sim_data)
    is_gt_const, is_sim_const = gt_std < const_tol, sim_std < const_tol
    if is_gt_const and is_sim_const:
        return 0.0 if np.isclose(np.mean(gt_data), np.mean(sim_data), atol=const_tol) else 1.0
    combined_data = np.concatenate([gt_data, sim_data])
    min_val, max_val = np.min(combined_data), np.max(combined_data)
    margin = (max_val - min_val) * 0.1
    x_grid = np.linspace(min_val - margin, max_val + margin, n_points).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian')
    kde.fit(gt_data.reshape(-1, 1))
    p_vals = np.exp(kde.score_samples(x_grid))
    kde.fit(sim_data.reshape(-1, 1))
    q_vals = np.exp(kde.score_samples(x_grid))
    epsilon = 1e-10
    P = p_vals / (p_vals.sum() + epsilon)
    Q = q_vals / (q_vals.sum() + epsilon)
    M = 0.5 * (P + Q)
    jsd_val = 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))
    return np.clip(jsd_val, 0, 1)


def calculate_fdd_for_endpoints(endpoints):
    """
    根据给定的N个终点，计算该时刻的FDD值。

    Args:
        endpoints (np.ndarray): 一个形状为 (N, 2) 的数组，其中N是轨迹数。

    Returns:
        float: 计算出的FDD值。
    """
    N = endpoints.shape[0]

    if N < 2:
        return 0.0

    pair_indices = itertools.combinations(range(N), 2)

    distances = [
        np.linalg.norm(endpoints[u] - endpoints[v])
        for u, v in pair_indices
    ]

    fdd = np.mean(distances)

    return fdd


# def load_dataset(results_dir):
#     helper = ReconstructionHelper(results_dir)
#     env, _, _ = helper.rebuild_environment()
#     dataset = env.dataset
#     cache_path, CacheClass = dataset.cache_path, dataset.cache_class
#     scene_name_to_gt_map = {scene.name: scene for scene in dataset.scenes()}
#     return scene_name_to_gt_map, cache_path, CacheClass


def caculate_JSD(sim_vel_agent, agent_gt_df, dt):
    sim_acc_agent = np.diff(sim_vel_agent) / dt if len(sim_vel_agent) >= 2 else np.array([])
    sim_jerk_agent = np.diff(sim_acc_agent) / dt if len(sim_acc_agent) >= 2 else np.array([])

    gt_vel_vector_xy, gt_acc_vector_xy = agent_gt_df[['vx', 'vy']].to_numpy(), agent_gt_df[['ax', 'ay']].to_numpy()
    gt_vel, gt_acc = np.linalg.norm(gt_vel_vector_xy, axis=1), np.linalg.norm(gt_acc_vector_xy, axis=1)
    gt_jerk = np.diff(gt_acc) / dt if len(gt_acc) >= 2 else np.array([])

    len_vel = min(len(gt_vel), len(sim_vel_agent))
    gt_vel_aligned, sim_vel_aligned = gt_vel[:len_vel], sim_vel_agent[:len_vel]

    len_acc = min(len(gt_acc), len(sim_acc_agent))
    gt_acc_aligned, sim_acc_aligned = gt_acc[:len_acc], sim_acc_agent[:len_acc]

    len_jerk = min(len(gt_jerk), len(sim_jerk_agent))
    gt_jerk_aligned, sim_jerk_aligned = gt_jerk[:len_jerk], sim_jerk_agent[:len_jerk]

    # 计算JSD
    jsd_vel = calculate_jsd_kde(gt_vel_aligned, sim_vel_aligned) if len_vel > 0 else np.nan
    jsd_acc = calculate_jsd_kde(gt_acc_aligned, sim_acc_aligned) if len_acc > 0 else np.nan
    jsd_jerk = calculate_jsd_kde(gt_jerk_aligned, sim_jerk_aligned) if len_jerk > 0 else np.nan

    return jsd_vel, jsd_acc, jsd_jerk


def caculate_FDD(agent_predictions):
    if agent_predictions.ndim != 4:
        return np.nan

    total_frames, num_predicted_trajectories, _, _ = agent_predictions.shape

    # 用于存储当前场景内，每个时刻的FDD值
    fdd_values_for_this_scene = []

    # 遍历场景中的每一个时间步 t
    for t in range(total_frames):
        trajectories_at_t = agent_predictions[t]

        endpoints_at_t = trajectories_at_t[:, -1, :]

        fdd_t = calculate_fdd_for_endpoints(endpoints_at_t)

        fdd_values_for_this_scene.append(fdd_t)

    avg_fdd_for_scene = np.mean(fdd_values_for_this_scene)

    return avg_fdd_for_scene


def get_vehicle_corners(pos, yaw, extent):
    """
    Get the four corners of a vehicle given its position, yaw, and extent

    Args:
        pos: [x, y] position of vehicle center
        yaw: yaw angle in radians
        extent: [length, width, height] of vehicle

    Returns:
        corners: array of shape (4, 2) containing [x, y] coordinates of corners
    """
    length, width = extent[0], extent[1]
    half_length, half_width = length / 2, width / 2

    # Define corners in vehicle's local coordinate system (relative to center)
    local_corners = np.array([
        [half_length, half_width],  # front right
        [half_length, -half_width],  # front left
        [-half_length, -half_width],  # rear left
        [-half_length, half_width]  # rear right
    ])

    # Rotation matrix
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    # Transform corners to global coordinate system
    global_corners = np.dot(local_corners, rotation_matrix.T) + pos

    return global_corners


def check_collision(pos_i, pos_j, yaw_i, yaw_j, extent_i, extent_j):
    """
    Check if two agents are in collision using oriented bounding boxes

    Args:
        pos_i, pos_j: positions [x, y]
        yaw_i, yaw_j: yaw angles in radians
        extent_i, extent_j: extents [length, width, height]

    Returns:
        bool: True if collision detected
    """
    # Get corners of both vehicles
    corners_i = get_vehicle_corners(pos_i, yaw_i, extent_i)
    corners_j = get_vehicle_corners(pos_j, yaw_j, extent_j)

    # Check if the oriented bounding boxes intersect
    return polygons_intersect(corners_i, corners_j)


def polygons_intersect(poly1, poly2):
    """
    Check if two polygons intersect using Separating Axes Theorem

    Args:
        poly1, poly2: arrays of shape (n, 2) containing polygon vertices

    Returns:
        bool: True if polygons intersect
    """
    # Check if any vertex of poly1 is inside poly2
    for point in poly1:
        if point_in_polygon(point, poly2):
            return True

    # Check if any vertex of poly2 is inside poly1
    for point in poly2:
        if point_in_polygon(point, poly1):
            return True

    # Check if any edges intersect
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        for j in range(n2):
            if line_segments_intersect(poly1[i], poly1[(i + 1) % n1],
                                       poly2[j], poly2[(j + 1) % n2]):
                return True

    return False


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm

    Args:
        point: [x, y] coordinates of the point
        polygon: array of shape (n, 2) containing polygon vertices

    Returns:
        bool: True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_segments_intersect(p1, q1, p2, q2):
    """
    Check if two line segments intersect

    Args:
        p1, q1: endpoints of first line segment
        p2, q2: endpoints of second line segment

    Returns:
        bool: True if segments intersect
    """

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0  # collinear
        return 1 if val > 0 else 2  # clockwise or counterclockwise

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    if (o1 == 0 and on_segment(p1, p2, q1)) or \
            (o2 == 0 and on_segment(p1, q2, q1)) or \
            (o3 == 0 and on_segment(p2, p1, q2)) or \
            (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False


def velocity_from_positions(positions, dt=0.1):
    """
    Calculate velocity from position data using finite differences

    Args:
        positions: array of shape (T, 2) containing [x, y] positions
        dt: time step interval

    Returns:
        velocities: array of shape (T, 2) containing [vx, vy] velocities
    """
    velocities = np.zeros_like(positions)

    # Forward difference for first point
    if len(positions) > 1:
        velocities[0] = (positions[1] - positions[0]) / dt

    # Central difference for middle points
    for t in range(1, len(positions) - 1):
        velocities[t] = (positions[t + 1] - positions[t - 1]) / (2 * dt)

    # Backward difference for last point
    if len(positions) > 1:
        velocities[-1] = (positions[-1] - positions[-2]) / dt

    return velocities


def calculate_ttc_safe_sim_method(pos_i, vel_i, pos_j, vel_j):
    """
    Calculate Time-to-Collision using the Safe-Sim method from equations A2-A3

    Args:
        pos_i, pos_j: positions of agents i and j [x, y]
        vel_i, vel_j: velocities of agents i and j [vx, vy]

    Returns:
        ttc: time to collision (or closest approach)
        d_col: distance at closest approach
    """
    # Calculate relative position and velocity
    dx = pos_i[0] - pos_j[0]
    dy = pos_i[1] - pos_j[1]
    dvx = vel_i[0] - vel_j[0]
    dvy = vel_i[1] - vel_j[1]

    # Calculate dv_squared
    dv_squared = dvx ** 2 + dvy ** 2

    # If relative velocity is zero, no collision will occur
    if dv_squared < 1e-6:
        return float('inf'), np.sqrt(dx ** 2 + dy ** 2)

    # Calculate time to closest approach using equation A2
    dot_product = dvx * dx + dvy * dy
    t_col_tilde = -dot_product / dv_squared

    if t_col_tilde >= 0:
        # Calculate distance at closest approach using equation A3
        cross_product = dvx * dy - dvy * dx
        d_col_squared = (cross_product ** 2) / dv_squared
        d_col_tilde = np.sqrt(d_col_squared)
        return t_col_tilde, d_col_tilde
    else:
        # If t_col_tilde < 0, use current distance
        current_distance = np.sqrt(dx ** 2 + dy ** 2)
        return 0, current_distance


def get_vehicle_pixel_corners(world_pos, world_yaw, world_extent, raster_from_world):
    """
    Get vehicle corners in pixel coordinates

    Args:
        world_pos: [x, y] position in world coordinates
        world_yaw: yaw angle in radians
        world_extent: [length, width, height] of vehicle
        raster_from_world: 3x3 transformation matrix

    Returns:
        pixel_corners: array of shape (4, 2) containing pixel coordinates of corners
    """
    # Get corners in world coordinates
    world_corners = get_vehicle_corners(world_pos, world_yaw, world_extent)

    # Convert each corner to pixel coordinates
    pixel_corners = np.zeros_like(world_corners)
    for i, corner in enumerate(world_corners):
        pixel_corners[i] = world_to_pixel_coordinates(corner, raster_from_world)

    return pixel_corners


def world_to_pixel_coordinates(world_pos, raster_from_world):
    """
    Convert world coordinates to pixel coordinates using transformation matrix

    Args:
        world_pos: [x, y] position in world coordinates
        raster_from_world: 3x3 transformation matrix

    Returns:
        pixel_pos: [px, py] position in pixel coordinates
    """
    # Convert to homogeneous coordinates
    world_homogeneous = np.array([world_pos[0], world_pos[1], 1.0])

    # Apply transformation
    pixel_homogeneous = np.dot(raster_from_world, world_homogeneous)

    # Convert back to 2D coordinates
    pixel_pos = pixel_homogeneous[:2] / pixel_homogeneous[2]

    return pixel_pos


def check_offroad_status(pixel_corners, drivable_map):
    """
    Check if vehicle is offroad based on pixel corners and drivable map

    Args:
        pixel_corners: array of shape (4, 2) containing pixel coordinates of vehicle corners
        drivable_map: 224x224 binary map where 1 indicates drivable area

    Returns:
        is_offroad: True if vehicle is offroad
    """
    map_height, map_width = drivable_map.shape

    # Get bounding box of vehicle in pixel coordinates
    min_x = max(0, int(np.floor(np.min(pixel_corners[:, 0]))))
    max_x = min(map_width - 1, int(np.ceil(np.max(pixel_corners[:, 0]))))
    min_y = max(0, int(np.floor(np.min(pixel_corners[:, 1]))))
    max_y = min(map_height - 1, int(np.ceil(np.max(pixel_corners[:, 1]))))

    # If vehicle is completely outside the map, consider it offroad
    if min_x >= map_width or max_x < 0 or min_y >= map_height or max_y < 0:
        return True, 1.0

    # Count pixels within vehicle boundary
    total_pixels = 0
    offroad_pixels = 0

    # Check each pixel in the bounding box
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            # Check if pixel is inside vehicle polygon
            if point_in_polygon([px, py], pixel_corners):
                total_pixels += 1
                # Check if this pixel is not drivable (0 in drivable_map)
                if drivable_map[py, px] == 0:
                    offroad_pixels += 1

    if offroad_pixels == 0:
        return False
    else:
        return True


RSS_PARAMS = {
    "rho": 0.5, "a_max_accel": 2.0, "a_min_brake": 4.0, "a_max_brake": 4.0,
    "mu": 0.2, "a_lat_max_accel": 1.5, "a_lat_min_brake": 2.5,
}


def calculate_longitudinal_dmin(v_rear: float, v_front: float, params: dict) -> float:
    """计算纵向RSS最小安全距离"""
    rho, a_max_accel, a_min_brake, a_max_brake = params['rho'], params['a_max_accel'], params['a_min_brake'], params[
        'a_max_brake']
    dist_reaction = v_rear * rho + 0.5 * a_max_accel * rho ** 2
    v_rear_after_reaction = v_rear + rho * a_max_accel
    dist_brake_rear = (v_rear_after_reaction ** 2) / (2 * a_min_brake)
    dist_brake_front = (v_front ** 2) / (2 * a_max_brake)
    d_min = dist_reaction + dist_brake_rear - dist_brake_front
    return max(0, d_min)


def calculate_lateral_dmin(v1_lat: float, v2_lat: float, params: dict) -> float:
    """计算横向RSS最小安全距离"""
    rho, mu, a_lat_max_accel, a_lat_min_brake = params['rho'], params['mu'], params['a_lat_max_accel'], params[
        'a_lat_min_brake']

    def get_lateral_stop_dist(v_lat):
        dist_reaction = abs(v_lat) * rho + 0.5 * a_lat_max_accel * rho ** 2
        v_lat_after_reaction = abs(v_lat) + rho * a_lat_max_accel
        dist_brake = (v_lat_after_reaction ** 2) / (2 * a_lat_min_brake)
        return dist_reaction + dist_brake

    dist1 = get_lateral_stop_dist(v1_lat)
    dist2 = get_lateral_stop_dist(v2_lat)
    return mu + dist1 + dist2


def check_rss_pair(ego_pos, ego_yaw, ego_vel, ego_extent, other_pos, other_yaw, other_vel, other_extent, params):
    """
    检查两个agent之间的RSS违规情况

    Args:
        ego_pos, other_pos: [x, y] positions
        ego_yaw, other_yaw: yaw angles in radians
        ego_vel, other_vel: [vx, vy] velocities
        ego_extent, other_extent: [length, width, height] extents
        params: RSS parameters dictionary

    Returns:
        lon_violation: longitudinal RSS violation count (0 or 1)
        lat_violation: lateral RSS violation count (0 or 1)
    """
    LATERAL_CHECK_RANGE_M = 5.0
    lon_violation = 0
    lat_violation = 0

    cos_yaw, sin_yaw = np.cos(ego_yaw), np.sin(ego_yaw)
    ego_half_len, ego_half_width = ego_extent[0] / 2, ego_extent[1] / 2
    other_half_len, other_half_width = other_extent[0] / 2, other_extent[1] / 2

    # 计算相对位置（转换到ego的坐标系）
    rel_pos = other_pos - ego_pos
    x_ego_frame = rel_pos[0] * cos_yaw + rel_pos[1] * sin_yaw
    y_ego_frame = -rel_pos[0] * sin_yaw + rel_pos[1] * cos_yaw

    # 纵向检查（other在ego前方）
    if x_ego_frame > 0 and abs(y_ego_frame) < (ego_half_width + other_half_width):
        actual_gap = x_ego_frame - ego_half_len - other_half_len
        if actual_gap < 0:
            # 碰撞
            lon_violation = 1
        else:
            # 计算纵向速度
            ego_v_lon = np.hypot(ego_vel[0], ego_vel[1])
            other_v_lon = other_vel[0] * cos_yaw + other_vel[1] * sin_yaw

            # 计算RSS最小距离
            d_min_lon = calculate_longitudinal_dmin(v_rear=ego_v_lon, v_front=other_v_lon, params=params)

            if actual_gap < d_min_lon:
                lon_violation = 1

    # 横向检查（车辆横向对齐）
    is_laterally_aligned = abs(x_ego_frame) < (ego_half_len + other_half_len)
    is_laterally_close = abs(y_ego_frame) < (ego_half_width + other_half_width + LATERAL_CHECK_RANGE_M)

    if is_laterally_aligned and is_laterally_close:
        actual_lateral_gap = abs(y_ego_frame) - ego_half_width - other_half_width
        if actual_lateral_gap < 0:
            # 碰撞
            lat_violation = 1
        else:
            # 计算横向速度
            other_v_lat = -other_vel[0] * sin_yaw + other_vel[1] * cos_yaw

            # 计算RSS最小距离
            d_min_lat = calculate_lateral_dmin(v1_lat=0, v2_lat=other_v_lat, params=params)

            if actual_lateral_gap < d_min_lat:
                lat_violation = 1

    return lon_violation, lat_violation


def caculate_FDD_all_agents(all_agent_predictions):
    """
    计算所有agent的FDD并取均值

    Args:
        all_agent_predictions: shape (num_agents, total_frames, num_predicted_trajectories, pred_horizon, 2)

    Returns:
        avg_fdd_all_agents: 所有agent的FDD均值
    """
    if all_agent_predictions.ndim != 5:
        return np.nan

    num_agents = all_agent_predictions.shape[0]
    agent_fdd_values = []

    # 计算每个agent的FDD
    for agent_idx in range(num_agents):
        agent_predictions = all_agent_predictions[
            agent_idx]  # shape: (total_frames, num_predicted_trajectories, pred_horizon, 2)

        if agent_predictions.ndim != 4:
            continue

        total_frames, num_predicted_trajectories, _, _ = agent_predictions.shape

        # 用于存储当前agent每个时刻的FDD值
        fdd_values_for_agent = []

        # 遍历该agent的每一个时间步 t
        for t in range(total_frames):
            # 提取当前时刻t的所有预测轨迹
            trajectories_at_t = agent_predictions[t]  # shape: (N, Pred_Horizon, 2)

            # 提取所有轨迹的终点 (最后一个预测点)
            endpoints_at_t = trajectories_at_t[:, -1, :]  # shape: (N, 2)

            # 计算当前时刻的FDD
            fdd_t = calculate_fdd_for_endpoints(endpoints_at_t)
            fdd_values_for_agent.append(fdd_t)

        # 计算该agent的平均FDD
        if fdd_values_for_agent:
            avg_fdd_for_agent = np.mean(fdd_values_for_agent)
            agent_fdd_values.append(avg_fdd_for_agent)

    # 计算所有agent的FDD均值
    if agent_fdd_values:
        return np.mean(agent_fdd_values)
    else:
        return np.nan

if __name__ == "__main__":

    cjsd = True
    cfdd = True

    results_dir = "/home/visier/safety-critical/safety_critical_trained_models/ppo_latent_dm"
    output_csv = f"{results_dir}/evaluation_results.csv"

    # if cjsd:
    #     scene_name_to_gt_map, cache_path, CacheClass = load_dataset(results_dir)

    # 存储所有场景结果的列表
    results_list = []

    # agents_data_dir = f"{results_dir}/agent_data_copy.json"
    # with open(agents_data_dir) as f:
    #     agent_data = json.load(f)

    h5_filepath = f"{results_dir}/data.hdf5"
    with h5py.File(h5_filepath, 'r') as f:
        for i, scene_name_with_episode in enumerate(f.keys()):
            sim_group = f[scene_name_with_episode]
            sim_vel = sim_group['curr_speed']  # Shape: (agent, T)
            sim_yaw = sim_group['yaw']  # Shape: (agent, T)
            sim_pos = sim_group['centroid']  # Shape: (agent, T, 2)
            sim_extent = sim_group['extent']  # Shape: (agent, T, 3)
            raster_from_world = sim_group['raster_from_world'][:]  # Shape: (agent, T, 3, 3)
            drivable_map = sim_group['drivable_map'][:]  # Shape: (agent, T, 224, 224)
            all_agent_predictions = sim_group['action_sample_positions'][:] # Shape: (agent, T, 20, 32)
            #scene_idx = sim_group['scene_index'][:]
            #track_id = sim_group['track_id'][:]

            print(f"Process {i + 1}: {scene_name_with_episode}")

            num_agents, num_timesteps = sim_pos.shape[:2]

            scene_metrics = {
                'scene_name': scene_name_with_episode,
                'num_agents': num_agents,
                'min_ttc': float('inf'),
                'offroad_rates': [],
                'collision_rates': [],
                'rss_lon': 0,
                'rss_lat': 0,
                'jsd_vel_list': [],
                'jsd_acc_list': [],
                'jsd_jerk_list': [],
                'fdd': np.nan
            }

            agent_offroad_counts = [0] * num_agents
            agent_collision_counts = [0] * num_agents
            collision_occurred = [False] * num_agents

            # JSD
            if cjsd:
                base_scene_name = scene_name_with_episode.split('_ego_')[0] if '_ego_' in scene_name_with_episode else \
                scene_name_with_episode.split('_')[0]
                if base_scene_name in scene_name_to_gt_map:
                    scene_metadata = scene_name_to_gt_map[base_scene_name]
                    scene_cache_instance = CacheClass(cache_path, scene_metadata, augmentations=None)
                    agent_df = scene_cache_instance.scene_data_df.reset_index()
                    dt = scene_metadata.dt

                    for agent_idx in range(num_agents):
                        agent_name = agent_data[base_scene_name][agent_idx]
                        gt_df = agent_df[agent_df['agent_id'] == agent_name].sort_values('scene_ts')

                        if not gt_df.empty:
                            jsd_vel, jsd_acc, jsd_jerk = caculate_JSD(sim_vel[agent_idx, :], gt_df, dt)
                            if not np.isnan(jsd_vel):
                                scene_metrics['jsd_vel_list'].append(jsd_vel)
                            if not np.isnan(jsd_acc):
                                scene_metrics['jsd_acc_list'].append(jsd_acc)
                            if not np.isnan(jsd_jerk):
                                scene_metrics['jsd_jerk_list'].append(jsd_jerk)

            # FDD
            if cfdd:
                scene_metrics['fdd'] = caculate_FDD_all_agents(all_agent_predictions)

            dt_val = scene_metadata.dt

            for t in range(num_timesteps):
                velocities = {}
                for agent_idx in range(num_agents):
                    if t > 0:
                        vel_x = (sim_pos[agent_idx, t, 0] - sim_pos[agent_idx, t - 1, 0]) / dt_val
                        vel_y = (sim_pos[agent_idx, t, 1] - sim_pos[agent_idx, t - 1, 1]) / dt_val
                        velocities[agent_idx] = np.array([vel_x, vel_y])
                    else:
                        velocities[agent_idx] = np.array([0.0, 0.0])

                for agent_i in range(num_agents):
                    for agent_j in range(agent_i + 1, num_agents):
                        pos_i = sim_pos[agent_i, t]
                        pos_j = sim_pos[agent_j, t]
                        yaw_i = sim_yaw[agent_i, t]
                        yaw_j = sim_yaw[agent_j, t]
                        extent_i = sim_extent[agent_i, t]
                        extent_j = sim_extent[agent_j, t]
                        vel_i = velocities[agent_i]
                        vel_j = velocities[agent_j]

                        # 碰撞检测
                        if check_collision(pos_i, pos_j, yaw_i, yaw_j, extent_i, extent_j):
                            agent_collision_counts[agent_i] += 1
                            agent_collision_counts[agent_j] += 1
                            collision_occurred[agent_i] = True
                            collision_occurred[agent_j] = True

                        # TTC
                        if not any(collision_occurred):
                            ttc, d_col = calculate_ttc_safe_sim_method(pos_i, vel_i, pos_j, vel_j)
                            if 0 < ttc < scene_metrics['min_ttc']:
                                scene_metrics['min_ttc'] = ttc

                        # RSS
                        # agent_i作为ego，agent_j作为other
                        lon_viol_i, lat_viol_i = check_rss_pair(pos_i, yaw_i, vel_i, extent_i,
                                                                pos_j, yaw_j, vel_j, extent_j, RSS_PARAMS)
                        scene_metrics['rss_lon'] += lon_viol_i
                        scene_metrics['rss_lat'] += lat_viol_i

                        # agent_j作为ego，agent_i作为other
                        lon_viol_j, lat_viol_j = check_rss_pair(pos_j, yaw_j, vel_j, extent_j,
                                                                pos_i, yaw_i, vel_i, extent_i, RSS_PARAMS)
                        scene_metrics['rss_lon'] += lon_viol_j
                        scene_metrics['rss_lat'] += lat_viol_j

                if any(collision_occurred):
                    scene_metrics['min_ttc'] = 0

                # off road
                for agent_idx in range(num_agents):
                    if agent_idx < raster_from_world.shape[0] and agent_idx < drivable_map.shape[0]:
                        agent_pos = sim_pos[agent_idx, t]
                        agent_yaw = sim_yaw[agent_idx, t]
                        agent_extent = sim_extent[agent_idx, t]
                        agent_transform = raster_from_world[agent_idx, t]
                        agent_map = drivable_map[agent_idx, t]

                        pixel_corners = get_vehicle_pixel_corners(agent_pos, agent_yaw, agent_extent, agent_transform)
                        if check_offroad_status(pixel_corners, agent_map):
                            agent_offroad_counts[agent_idx] += 1


            for agent_idx in range(num_agents):
                offroad_rate = agent_offroad_counts[agent_idx] / num_timesteps
                collision_rate = agent_collision_counts[agent_idx] / num_timesteps
                scene_metrics['offroad_rates'].append(offroad_rate)
                scene_metrics['collision_rates'].append(collision_rate)

            scene_jsd_vel = np.mean(scene_metrics['jsd_vel_list']) if scene_metrics['jsd_vel_list'] else np.nan
            scene_jsd_acc = np.mean(scene_metrics['jsd_acc_list']) if scene_metrics['jsd_acc_list'] else np.nan
            scene_jsd_jerk = np.mean(scene_metrics['jsd_jerk_list']) if scene_metrics['jsd_jerk_list'] else np.nan

            final_metrics = {
                'scene_name': scene_metrics['scene_name'],
                'num_agents': scene_metrics['num_agents'],
                'min_ttc': scene_metrics['min_ttc'] if scene_metrics['min_ttc'] != float('inf') else np.nan,
                'avg_offroad_rate': np.mean(scene_metrics['offroad_rates']),
                'avg_collision_rate': np.mean(scene_metrics['collision_rates']),
                'rss_lon': scene_metrics['rss_lon'],
                'rss_lat': scene_metrics['rss_lat'],
                'avg_jsd_vel': scene_jsd_vel,
                'avg_jsd_acc': scene_jsd_acc,
                'avg_jsd_jerk': scene_jsd_jerk,
                'fdd': scene_metrics['fdd']
            }

            results_list.append(final_metrics)

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_csv, index=False)
        print(f"\nEvaluation results saved to: {output_csv}")
        print(f"Total processed: {len(results_df)} scenarios")
    else:
        print("No results to save.")