"""Visualize model output."""

import numpy as np
import random
import torch
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from Models.Proposed import DynamicRadioMapNet
from Models.baselines import ConvLSTMRadioMap, Full3DUNet
from Models.RadioUNet import RadioWNet
from Utils.create_gif import create_gif, create_combined_gif

warnings.filterwarnings('ignore')


def hard_thresholding(x, thresh=0.2):
    mask = x < thresh
    x[mask] = thresh
    x = x - thresh * np.ones(np.shape(x))
    x = x / (1 - thresh)
    return x

def compute_nmse(x, x_hat):
    denominator = np.linalg.norm(x.ravel()) ** 2
    diff = x_hat - x
    numerator = np.linalg.norm(diff.ravel()) ** 2
    nmse = numerator / denominator
    return nmse


# --- Define paths and parameters ---
dataset_dir = '/home/user/datasets/DynamicRadioMap'
model_proposed_path = 'Checkpoints/Proposed/best.pth'
model_lstm_path = 'Checkpoints/ConvLSTM/best.pth'
model_unet3d_path = 'Checkpoints/Full3DUNet/best.pth'
model_radiounet_path = 'Checkpoints/RadioUNet/best.pth'
device = 'cuda:3'
cmap = 'viridis'

# --- Get normalization stats ---
dataset_stats = {
    'building_height_max': 40.0,
    'area_size': 128.0,
    'threshold': -110.0,
    'global_rm_max': -83.9375,
    'global_rm_mean': 0.46969,
    'global_rm_std': 0.37251,
}

# --- Load checkpoint ---
model_proposed_checkpoint = torch.load(model_proposed_path, map_location=device)
model_lstm_checkpoint = torch.load(model_lstm_path, map_location=device)
model_unet3d_checkpoint = torch.load(model_unet3d_path, map_location=device)
model_radiounet_checkpoint = torch.load(model_radiounet_path, map_location=device)

# --- Create model ---
model_proposed = DynamicRadioMapNet(
    scene_widths=(32, 64, 128, 256, 256),
    control_widths=(8, 16, 24, 32, 48),
    film_hidden=128,
    temporal_depth=2,
    temporal_kernel=5,
    tx_sigma_px=1.5,
).to(device)
model_lstm = ConvLSTMRadioMap(
    widths=(48, 96, 192, 192),
    hidden_ch=256,
    convlstm_kernel=3,
    use_prev_frame=False,
).to(device)
model_unet3d = Full3DUNet(widths=(32, 64, 128, 192, 192)).to(device)
model_radiounet = RadioWNet(inputs=2, phase="secondU").to(device)

# --- Load model weights ---
model_proposed.load_state_dict(model_proposed_checkpoint["model"])
model_lstm.load_state_dict(model_lstm_checkpoint["model"])
model_unet3d.load_state_dict(model_unet3d_checkpoint["model"])
model_radiounet.load_state_dict(model_radiounet_checkpoint["model"])

# --- Set to eval mode ---
model_proposed.eval()
model_lstm.eval()
model_unet3d.eval()
model_radiounet.eval()

# --- Randomly select a sample for visualization ---
idx_file = "Utils/test_idx.npy"
idx_list = np.load(idx_file)
total_samples = idx_list.shape[0]
selected_idx = random.randint(0, total_samples - 1)
tile_idx = idx_list[selected_idx, 0]
subtile_idx = idx_list[selected_idx, 1]
traj_idx = idx_list[selected_idx, 2]
# tile_idx = 4
# subtile_idx = 46
# traj_idx = 24
print('Randomly selected tile index:', tile_idx)
print('Randomly selected sub-tile index:', subtile_idx)
print('Randomly selected trajectory index:', traj_idx)

# --- Load building, trajectory, and ground truth radio map ---
building_file = os.path.join(dataset_dir, 'buildings', f'{tile_idx}_{subtile_idx}.npy')
traj_file = os.path.join(dataset_dir, 'trajs_array.npy')
radio_file = os.path.join(dataset_dir, 'raw_radio_maps', f'{tile_idx}_{subtile_idx}_{traj_idx}.npy')

building_map = np.load(building_file)                      # (128, 128)
trajs_array = np.load(traj_file)                           # (5, 64, 30, 64, 2)
trajectory = trajs_array[tile_idx, subtile_idx, traj_idx]  # (64, 2)
radio_map = np.load(radio_file)                            # (64, 128, 128)

# --- Create transmitter locatio map ---
num_waypoints = trajectory.shape[0]
tx_loc_map = np.zeros_like(radio_map, dtype=np.float32)          # (64, 128, 128)
for waypoint_idx in range(num_waypoints):
    radio_map_p = radio_map[waypoint_idx]                        # (128, 128)
    tx_loc_map_p = np.zeros_like(radio_map_p, dtype=np.float32)  # (128, 128)
    max_idx = np.unravel_index(np.argmax(radio_map_p), radio_map_p.shape)
    tx_loc_map_p[max_idx] = 1.0
    tx_loc_map[waypoint_idx] = tx_loc_map_p

# --- Normalize inputs ---
building_map = building_map / dataset_stats['building_height_max']  # Normalize building heights to [0, 1]
trajectory = trajectory / dataset_stats['area_size']                # Normalize (x, y) to [0, 1]

# --- Normalize & standardize output ---
threshold = dataset_stats['threshold']
global_radio_map_max = dataset_stats['global_rm_max']
global_radio_map_mean = dataset_stats['global_rm_mean']
global_radio_map_std = dataset_stats['global_rm_std']
radio_map[radio_map <= threshold] = threshold                             # Clip to threshold
radio_map = (radio_map - threshold) / (global_radio_map_max - threshold)  # Scale to [0, 1]
radio_map = (radio_map - global_radio_map_mean) / global_radio_map_std    # Standardize to mean=0, std=1

# --- Convert to tensors ---
building_map_t = torch.from_numpy(building_map).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 128, 128)
trajectory_t = torch.from_numpy(trajectory).unsqueeze(0).float().to(device)                   # (1, 64, 2)
tx_loc_map_t = torch.from_numpy(tx_loc_map).unsqueeze(0).float().to(device)                   # (1, 64, 128, 128)
radio_map_t = torch.from_numpy(radio_map).unsqueeze(1).unsqueeze(0).float().to(device)        # (1, 64, 1, 128, 128)

# --- Test model ---
with torch.no_grad():
    pred_proposed = model_proposed(building_map_t, trajectory_t)          # (1, 64, 1, 128, 128)
    pred_lstm = model_lstm(building_map_t, trajectory_t)                  # (1, 64, 1, 128, 128)
    pred_unet3d = model_unet3d(building_map_t, trajectory_t)              # (1, 64, 1, 128, 128)

    pred_radiounet = torch.zeros_like(radio_map_t).to(device)             # (1, 64, 1, 128, 128)
    for waypoint_idx in range(pred_proposed.shape[1]):
        tx_loc_map_p = tx_loc_map_t[0, waypoint_idx]                      # (128, 128)
        tx_loc_map_p.unsqueeze_(0).unsqueeze_(0)                          # (1, 1, 128, 128)
        inputs = torch.cat([building_map_t, tx_loc_map_p], dim=1)  # (1, 2, 128, 128)
        [_, pred_radiounet_p] =model_radiounet(inputs)                    # (1, 1, 128, 128)
        pred_radiounet[:, waypoint_idx] = pred_radiounet_p


# --- Move to CPU ---
pred_proposed = pred_proposed.cpu().numpy()    # (1, 64, 1, 128, 128)
pred_lstm = pred_lstm.cpu().numpy()            # (1, 64, 1, 128, 128)
pred_unet3d = pred_unet3d.cpu().numpy()        # (1, 64, 1, 128, 128)
pred_radiounet = pred_radiounet.cpu().numpy()  # (1, 64, 1, 128, 128)

# --- Destandardize ---
pred_proposed = pred_proposed * global_radio_map_std + global_radio_map_mean
pred_proposed = np.clip(pred_proposed, 0, 1)
pred_lstm = pred_lstm * global_radio_map_std + global_radio_map_mean
pred_lstm = np.clip(pred_lstm, 0, 1)
pred_unet3d = pred_unet3d * global_radio_map_std + global_radio_map_mean
pred_unet3d = np.clip(pred_unet3d, 0, 1)
pred_radiounet = np.clip(pred_radiounet, 0, 1)

radio_map = radio_map * global_radio_map_std + global_radio_map_mean  # Denormalize
radio_map = np.clip(radio_map, 0, 1)

pred_proposed = pred_proposed.squeeze()    # (64, 128, 128)
pred_lstm = pred_lstm.squeeze()            # (64, 128, 128)
pred_unet3d = pred_unet3d.squeeze()        # (64, 128, 128)
pred_radiounet = pred_radiounet.squeeze()  # (64, 128, 128)

# --- Hard thresholding ---
pred_proposed = hard_thresholding(pred_proposed, thresh=0.0)
pred_lstm = hard_thresholding(pred_lstm, thresh=0.0)
pred_unet3d = hard_thresholding(pred_unet3d, thresh=0.0)
pred_radiounet = hard_thresholding(pred_radiounet, thresh=0.0)
radio_map = hard_thresholding(radio_map, thresh=0.0)

# --- Compute Normalized MSE ---
nmse_proposed = compute_nmse(radio_map, pred_proposed)
nmse_lstm = compute_nmse(radio_map, pred_lstm)
nmse_unet3d = compute_nmse(radio_map, pred_unet3d)
nmse_radiounet = compute_nmse(radio_map, pred_radiounet)
print(f"NMSE of Prediction Proposed : {nmse_proposed:.4f}")
print(f"NMSE of Prediction ConvLSTM : {nmse_lstm:.4f}")
print(f"NMSE of Prediction UNet3D   : {nmse_unet3d:.4f}")
print(f"NMSE of Prediction RadioUNet: {nmse_radiounet:.4f}")

# --- Visualize the ground truth and predictions ---
font_size = 30
col_labels = ['(a) Ground Truth', '(b) Proposed', '(c) ConvLSTM', '(d) UNet3D', '(e) RadioUNet']
num_models = len(col_labels)
row_labels = ['$t = 1$', '$t = 16$', '$t = 32$', '$t = 48$', '$t = 64$']
time_steps = [0, 15, 31, 47, 63]
num_steps = len(time_steps)
data_list = [radio_map, pred_proposed, pred_lstm, pred_unet3d, pred_radiounet]

# Trajectory coordinates (pixel space, 0–127)
traj_x = trajs_array[tile_idx, subtile_idx, traj_idx, :, 0]
traj_y = trajs_array[tile_idx, subtile_idx, traj_idx, :, 1]

fig = plt.figure(figsize=(28, 25))
gs = gridspec.GridSpec(num_steps, num_models + 1, figure=fig,
                       width_ratios=[1, 1, 1, 1, 1, 0.06],
                       hspace=0.06, wspace=0.06,
                       left=0.10, right=0.93, top=0.95, bottom=0.03)

axs = [[fig.add_subplot(gs[row, col]) for col in range(num_models)] for row in range(num_steps)]
cbar_ax = fig.add_subplot(gs[:, num_models])

im = None
for row in range(num_steps):
    t = time_steps[row]
    for col in range(num_models):
        ax = axs[row][col]
        im = ax.imshow(data_list[col][t, :, :], cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Full trajectory (white dashed line)
        ax.plot(traj_x, traj_y, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
        # Current waypoint highlight (cyan dot with red outline)
        ax.plot(traj_x[t], traj_y[t], 'o', color='cyan', markersize=10, zorder=5,
                markeredgecolor='red', markeredgewidth=1.5)

        # Legend (shown on every subplot)
        legend_handles = [
            Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', alpha=0.8, label='Trajectory'),
            Line2D([0], [0], marker='o', color='r', markerfacecolor='cyan',
                   markeredgecolor='red', markeredgewidth=1.5, markersize=8, label='Current Location'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=font_size - 14, framealpha=0.6)

        # Column labels on first row only
        if row == 0:
            ax.set_title(col_labels[col], fontsize=font_size + 2, pad=14, fontweight='bold')

        # Row labels on first column only
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=font_size + 2, rotation=0,
                          labelpad=55, va='center', ha='right', fontweight='bold')

# Single colorbar spanning all 5 rows
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Normalized Path Loss', fontsize=font_size, labelpad=15)
cbar.ax.tick_params(labelsize=font_size - 2)

os.makedirs('SavedImages_no_perc/snapshots', exist_ok=True)
plt.savefig(f'SavedImages_no_perc/snapshots/snapshot_{tile_idx}_{subtile_idx}_{traj_idx}.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Animate and save as GIF ---
gif_fps = 2

gt_gif = create_gif(radio_map, cmap, traj_x, traj_y, gif_fps, save=False, save_filename=None)
pred_proposed_gif = create_gif(pred_proposed, cmap, traj_x, traj_y, gif_fps, save=False, save_filename=None)
pred_lstm_gif = create_gif(pred_lstm, cmap, traj_x, traj_y, gif_fps, save=False, save_filename=None)
pred_unet3d_gif = create_gif(pred_unet3d, cmap, traj_x, traj_y, gif_fps, save=False, save_filename=None)
pred_radiounet_gif = create_gif(pred_radiounet, cmap, traj_x, traj_y, gif_fps, save=False, save_filename=None)

# --- Combined GIF ---
combined_gif = create_combined_gif(
    radio_maps=[radio_map, pred_proposed, pred_lstm, pred_unet3d, pred_radiounet],
    titles=['Ground Truth', 'Proposed', 'ConvLSTM', 'UNet3D', 'RadioUNet'],
    cmap=cmap,
    traj_x=traj_x, traj_y=traj_y,
    gif_fps=gif_fps, save=True,
    save_filename=f"SavedImages_no_perc/dynamic_radio_map_{tile_idx}_{subtile_idx}_{traj_idx}.gif"
)

