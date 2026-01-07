# generate.py
import datetime
import os

import matplotlib.pyplot as plt
import torch
from config import config
from diffusers import DDPMScheduler
from diffusion_module import DiffusionExperiment
from tqdm import tqdm

from config import config

# 加载训练好的模型
model = DiffusionExperiment.load_from_checkpoint(
    f"{config.save_dir}/best-checkpoint.ckpt"
)
print(f"模型加载成功: {config.save_dir}/best-checkpoint.ckpt")
print(f"采样类型: {model.prediction_type}")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 准备调度器
noise_scheduler = DDPMScheduler.from_config(
    {"num_train_timesteps": config.num_train_timesteps}
)

# noise_scheduler = model.noise_scheduler

num_images = 16
save_path = f"{config.save_dir}/generated_images_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

# 生成图像
with torch.no_grad():
    sample = torch.randn(
        num_images, 1, config.image_size, config.image_size, device=model.device
    )

    for t in tqdm(noise_scheduler.timesteps):
        if model.prediction_type == "epsilon":
            residual = model.model(sample, t).sample
            sample = noise_scheduler.step(residual, t, sample)["prev_sample"]

        elif model.prediction_type == "v_prediction":
            pass

        elif model.prediction_type == "sample":
            image_pred = model.model(sample, t).sample
            # 获取当前时间步的 sqrt_alphas_cumprod 和 sqrt_one_minus_alphas_cumprod
            # 与训练/验证过程保持一致
            t_idx = t.item() if isinstance(t, torch.Tensor) else t
            sqrt_alphas_cumprod_t = noise_scheduler.sqrt_alphas_cumprod.to(
                sample.device
            )[t_idx]
            sqrt_one_minus_alphas_cumprod_t = (
                noise_scheduler.sqrt_one_minus_alphas_cumprod.to(sample.device)[t_idx]
            )
            # 调整维度以便广播
            for _ in range(sample.ndim - 1):
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod_t = (
                    sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
                )
            # 将图像预测转换为噪声预测（与训练/验证过程一致）
            noise_pred = (
                sample - sqrt_alphas_cumprod_t * image_pred
            ) / sqrt_one_minus_alphas_cumprod_t
            sample = noise_scheduler.step(noise_pred, t, sample)["prev_sample"]

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
