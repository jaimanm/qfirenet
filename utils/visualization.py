import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def plot_training_history(hist, save_path):
    """Plot training loss, OA, mIoU, and time per batch."""
    seg_losses, oas, mious, times = zip(*hist)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(seg_losses, label='Segmentation Loss')
    axs[0, 0].set_xlabel('Batch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Segmentation Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(oas, label='Overall Accuracy')
    axs[0, 1].set_xlabel('Batch')
    axs[0, 1].set_ylabel('OA')
    axs[0, 1].set_title('Overall Accuracy')
    axs[0, 1].legend()

    axs[1, 0].plot(mious, label='Mean IoU')
    axs[1, 0].set_xlabel('Batch')
    axs[1, 0].set_ylabel('mIoU')
    axs[1, 0].set_title('Mean IoU')
    axs[1, 0].legend()

    axs[1, 1].plot(times, label='Time per Batch')
    axs[1, 1].set_xlabel('Batch')
    axs[1, 1].set_ylabel('Time (s)')
    axs[1, 1].set_title('Time per Batch')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scene_map(reconstructed_rgb, reconstructed_pred, reconstructed_label, save_path):
    """Plot RGB, detection overlay, and ground truth label side by side."""
    cmap = ListedColormap(['white', 'red'])
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(reconstructed_rgb.transpose(1, 2, 0) / 1500.)
    axs[0].axis('off')
    axs[0].set_title('RGB image', fontsize=12)

    axs[1].imshow(reconstructed_rgb.transpose(1, 2, 0) / 1500., alpha=0.6)
    axs[1].imshow(reconstructed_pred, cmap=cmap, alpha=0.7)
    axs[1].axis('off')
    axs[1].set_title('Detection', fontsize=12)

    axs[2].imshow(reconstructed_rgb.transpose(1, 2, 0) / 1500., alpha=0.6)
    axs[2].imshow(reconstructed_label, cmap=cmap, alpha=0.7)
    axs[2].axis('off')
    axs[2].set_title('Label', fontsize=12)

    legend_labels = ['Non-fire', 'Fire']
    plt.legend(
        handles=[Rectangle((0, 0), 1, 1, facecolor=cmap(i), edgecolor='black') for i in range(2)],
        labels=legend_labels, fontsize=12, frameon=False,
        bbox_to_anchor=(1.04, 0), loc="lower left"
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_heatmap(prob_map, title, save_path):
    """Plot a continuous fire probability map with colorbar."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('fire probability', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
