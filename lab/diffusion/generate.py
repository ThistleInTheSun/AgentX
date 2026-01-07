# generate.py
import matplotlib.pyplot as plt
import torch
from config import config
from diffusers import DDPMScheduler
from diffusion_module import DiffusionExperiment
from tqdm import tqdm
import datetime
import os

# 加载训练好的模型
model = DiffusionExperiment.load_from_checkpoint("./results/best-checkpoint.ckpt")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 准备调度器
noise_scheduler = DDPMScheduler.from_config(
    {"num_train_timesteps": config.num_train_timesteps}
)

num_images = 16
save_path = (
    f"./generated_images_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
)

# 生成图像
with torch.no_grad():
    sample = torch.randn(
        num_images, 1, config.image_size, config.image_size, device=model.device
    )

    for t in tqdm(noise_scheduler.timesteps):
        residual = model.model(sample, t).sample
        sample = noise_scheduler.step(residual, t, sample).prev_sample

    sample = (sample + 1) / 2  # 反归一化

# 显示结果
# 计算网格大小（尽量接近正方形）
grid_size = int(num_images**0.5)
if grid_size * grid_size < num_images:
    grid_size += 1

fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
axes = axes.flatten() if num_images > 1 else [axes]

for i, ax in enumerate(axes):
    if i < num_images:
        # 显示单通道灰度图像
        img = sample[i, 0].cpu().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.suptitle(
    f"生成的图像 (推理步数: {len(noise_scheduler.timesteps)}, 训练步数: {config.num_train_timesteps})",
    fontsize=14,
)
plt.tight_layout()

# 保存图像
if save_path:
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图像已保存到: {save_path}")

plt.show()
