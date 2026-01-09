# diffusion_module.py
import datetime
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from config import config
from diffusers import UNet2DModel
from scheduler import DDPMScheduler
from torch.optim import Adam


class DiffusionExperiment(pl.LightningModule):
    def __init__(self, prediction_type):
        super().__init__()

        self.prediction_type = prediction_type  #  "epsilon" | "jit"

        # 1. 初始化UNet模型（后续可轻松替换为ViT或其他）
        self.model = UNet2DModel(**config.model_config)

        # 2. 初始化扩散过程的噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
        )

        # 保存超参数，便于日志记录
        self.save_hyperparameters()

    def forward(self, x, timesteps):
        # 前向传播：输入含噪声图像和时间步，预测噪声
        return self.model(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        # 训练步骤
        images, _ = batch  # 忽略标签

        # 生成随机噪声
        noise = torch.randn_like(images)

        # 随机采样时间步
        timesteps = torch.randint(
            0,
            config.num_train_timesteps,
            (images.shape[0],),
            device=self.device,
        ).long()

        # 根据时间步向图像添加噪声（前向扩散过程）
        if self.prediction_type == "epsilon":
            noisy_images, _, _ = self.noise_scheduler.add_noise(
                images, noise, timesteps
            )
            # 使用模型预测噪声
            noise_pred = self.model(noisy_images, timesteps).sample

        elif self.prediction_type == "jit":
            noisy_images, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t = (
                self.noise_scheduler.add_noise(images, noise, timesteps)
            )

            # 使用模型预测噪声
            image_pred = self.model(noisy_images, timesteps).sample

            noise_pred = (
                noisy_images - sqrt_alphas_cumprod_t * image_pred
            ) / sqrt_one_minus_alphas_cumprod_t

        # 计算简单的MSE损失
        loss = F.mse_loss(noise_pred, noise)

        # 记录日志
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证步骤（与训练步骤类似，但模型处于eval模式）
        images, _ = batch
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            config.num_train_timesteps,
            (images.shape[0],),
            device=self.device,
        ).long()

        if self.prediction_type == "epsilon":
            noisy_images, _, _ = self.noise_scheduler.add_noise(
                images, noise, timesteps
            )
            with torch.no_grad():
                noise_pred = self.model(noisy_images, timesteps).sample

        elif self.prediction_type == "jit":
            noisy_images, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t = (
                self.noise_scheduler.add_noise(images, noise, timesteps)
            )

            with torch.no_grad():
                image_pred = self.model(noisy_images, timesteps).sample

                noise_pred = (
                    noisy_images - sqrt_alphas_cumprod_t * image_pred
                ) / sqrt_one_minus_alphas_cumprod_t

        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 配置优化器
        optimizer = Adam(self.parameters(), lr=config.lr)
        return optimizer

    def on_train_epoch_end(self):
        # 每个训练周期结束时，生成示例图片以监控进度
        if self.current_epoch % 5 == 0:  # 每5个epoch生成一次
            self.generate_and_save_samples()

    def generate_and_save_samples(self):
        # 使用训练好的模型生成样本
        self.model.eval()

        with torch.no_grad():
            # 从随机噪声开始
            sample = torch.randn(
                16, 1, config.image_size, config.image_size, device=self.device
            )

            # 逐步去噪（反向扩散过程）
            for t in self.noise_scheduler.timesteps:
                if self.prediction_type == "epsilon":
                    residual = self.model(sample, t).sample
                    sample = self.noise_scheduler.step(residual, t, sample)["prev_sample"]

                elif self.prediction_type == "jit":
                    image_pred = self.model(sample, t).sample
                    # 获取当前时间步的 sqrt_alphas_cumprod 和 sqrt_one_minus_alphas_cumprod
                    # 与训练/验证过程一致
                    t_idx = t.item() if isinstance(t, torch.Tensor) else t
                    sqrt_alphas_cumprod_t = self.noise_scheduler.sqrt_alphas_cumprod.to(
                        sample.device
                    )[t_idx]
                    sqrt_one_minus_alphas_cumprod_t = (
                        self.noise_scheduler.sqrt_one_minus_alphas_cumprod.to(
                            sample.device
                        )[t_idx]
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
                    sample = self.noise_scheduler.step(noise_pred, t, sample)["prev_sample"]

            # 将生成结果从[-1,1]转换回[0,1]范围
            sample = (sample + 1) / 2

        # 保存图片到文件
        grid = torchvision.utils.make_grid(sample.cpu(), nrow=4)
        
        # 保存到文件
        save_dir = config.save_dir
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            save_dir, f"generated_images_epoch_{self.current_epoch}_{timestamp}.png"
        )
        torchvision.utils.save_image(grid, save_path)
        
        # 同时保存到 TensorBoard（如果 logger 可用）
        if self.logger is not None:
            try:
                self.logger.experiment.add_image(
                    "generated_images", grid, self.current_epoch
                )
            except Exception:
                pass  # 如果 TensorBoard 不可用，忽略错误

        self.model.train()
