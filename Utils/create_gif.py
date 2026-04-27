"""Generate GIF image for a dynamic radio map"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def create_gif(radio_map, cmap, traj_x, traj_y, gif_fps, save, save_filename):
    num_frames = radio_map.shape[0]
    interval_ms = int(np.clip(1000 / gif_fps, 200, 500))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(radio_map[0], cmap=cmap, vmin=0, vmax=1, animated=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Path Loss')

    # Static trajectory line (drawn once, always visible)
    ax.plot(traj_x, traj_y, color='white', linewidth=1.2, linestyle='--', alpha=0.8, label='Trajectory')

    # Animated current-waypoint marker
    wp_dot, = ax.plot(traj_x[0], traj_y[0], 'o', color='cyan', markersize=8,
                      zorder=5, label='Current waypoint')

    ax.legend(loc='upper right', fontsize=7)
    title = ax.set_title(f'Frame 1 / {num_frames}')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    fig.tight_layout()


    def update(frame_idx):
        im.set_data(radio_map[frame_idx])
        # Highlight the waypoint that corresponds to this frame
        wp_idx = min(frame_idx, len(traj_x) - 1)
        wp_dot.set_data([traj_x[wp_idx]], [traj_y[wp_idx]])
        title.set_text(f'Frame {frame_idx + 1} / {num_frames}  |  Waypoint {wp_idx + 1} / {len(traj_x)}')
        return im, wp_dot, title


    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=interval_ms, blit=True, repeat=False
    )

    if save:
        ani.save(save_filename, writer='pillow', fps=1000 // interval_ms)
        plt.close(fig)
    
    return ani


def create_combined_gif(radio_maps, titles, cmap, traj_x, traj_y, gif_fps, save, save_filename):
    """Combine multiple radio maps into a single side-by-side animated GIF (1 row x N cols)."""
    num_frames = radio_maps[0].shape[0]
    n_cols = len(radio_maps)
    interval_ms = int(np.clip(1000 / gif_fps, 200, 500))

    fig, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axs = [axs]

    ims, wp_dots, title_objs = [], [], []

    for ax, rm, title_str in zip(axs, radio_maps, titles):
        im = ax.imshow(rm[0], cmap=cmap, vmin=0, vmax=1, animated=True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Path Loss')
        ax.plot(traj_x, traj_y, color='white', linewidth=1.2, linestyle='--', alpha=0.8, label='Trajectory')
        wp_dot, = ax.plot(traj_x[0], traj_y[0], 'o', color='cyan', markersize=8, zorder=5, label='Current waypoint')
        ax.legend(loc='upper right', fontsize=7)
        t = ax.set_title(f'{title_str}  |  Frame 1 / {num_frames}')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        ims.append(im)
        wp_dots.append(wp_dot)
        title_objs.append(t)

    fig.tight_layout()

    def update(frame_idx):
        wp_idx = min(frame_idx, len(traj_x) - 1)
        artists = []
        for i in range(n_cols):
            ims[i].set_data(radio_maps[i][frame_idx])
            wp_dots[i].set_data([traj_x[wp_idx]], [traj_y[wp_idx]])
            title_objs[i].set_text(f'{titles[i]}  |  Frame {frame_idx + 1} / {num_frames}')
            artists.extend([ims[i], wp_dots[i], title_objs[i]])
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=interval_ms, blit=True, repeat=True
    )

    if save:
        ani.save(save_filename, writer='pillow', fps=1000 // interval_ms)
        plt.close(fig)

    return ani

