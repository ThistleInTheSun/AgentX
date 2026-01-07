from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


class DDPMScheduler:
    """
    完整的DDPM(Denoising Diffusion Probabilistic Models)噪声调度器
    实现了原始论文中的前向加噪和反向去噪过程
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
    ):
        """
        初始化DDPM调度器

        参数:
            num_train_timesteps: 扩散过程的总步数
            beta_start: beta序列的起始值
            beta_end: beta序列的结束值
            beta_schedule: beta调度类型 ["linear", "quadratic", "sigmoid", "cosine"]
            variance_type: 方差类型 ["fixed_small", "fixed_large", "learned", "learned_range"]
            clip_sample: 是否将样本裁剪到[-1, 1]
            prediction_type: 预测类型 ["epsilon"(噪声), "sample"(图像), "v_prediction"(速度)]
            thresholding: 是否使用动态阈值
            dynamic_thresholding_ratio: 动态阈值比率
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio

        # 初始化相关参数
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1).long()

        # 计算beta序列
        self.betas = self._get_beta_schedule()

        # 计算alpha序列
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # 计算sqrt参数（用于前向过程）
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 计算反向过程参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / self.alphas - 1)

        # 计算后验方差参数
        # 公式: \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # 计算后验均值的系数
        # 公式: \tilde{\mu}_t(x_t, x_0) =
        # \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0 +
        # \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # 计算log方差（用于学习方差的情况）
        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                [
                    torch.tensor([self.posterior_variance[1]]),
                    self.posterior_variance[1:],
                ]
            )
        )

    def _get_beta_schedule(self) -> torch.Tensor:
        """根据调度类型生成beta序列"""
        if self.beta_schedule == "linear":
            # 线性调度
            return torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                dtype=torch.float32,
            )
        elif self.beta_schedule == "quadratic":
            # 二次调度
            return (
                torch.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif self.beta_schedule == "sigmoid":
            # Sigmoid调度
            betas = torch.linspace(-6, 6, self.num_train_timesteps)
            return (
                torch.sigmoid(betas) * (self.beta_end - self.beta_start)
                + self.beta_start
            )
        elif self.beta_schedule == "cosine":
            # Cosine调度 (Improved DDPM)
            def alpha_bar(t):
                return torch.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

            timesteps = torch.linspace(
                0, self.num_train_timesteps, self.num_train_timesteps + 1
            )
            alphas_cumprod = alpha_bar(timesteps / self.num_train_timesteps)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向扩散过程：向原始图像添加噪声

        公式: x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon

        参数:
            original_samples: 原始图像 x_0
            noise: 随机噪声 \epsilon
            timesteps: 时间步

        返回:
            noisy_samples: 加噪后的图像 x_t
        """
        # 确保时间步在正确范围内
        timesteps = timesteps.long()

        # 获取对应时间步的sqrt(alpha_cumprod)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(original_samples.device)[
            timesteps
        ]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(
            original_samples.device
        )[timesteps]

        # 调整维度以便广播
        for _ in range(original_samples.ndim - 1):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(
                -1
            )

        # 应用前向扩散公式
        noisy_samples = (
            sqrt_alphas_cumprod_t * original_samples
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

        return noisy_samples, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算速度v（v-prediction parameterization）

        公式: v_t = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} x_0

        参数:
            sample: 图像 x_t
            noise: 噪声 \epsilon
            timesteps: 时间步

        返回:
            velocity: 速度 v_t
        """
        timesteps = timesteps.long()

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(sample.device)[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(
            sample.device
        )[timesteps]

        for _ in range(sample.ndim - 1):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(
                -1
            )

        velocity = (
            sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * sample
        )

        return velocity

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> dict:
        """
        单步反向扩散过程：从x_t预测x_{t-1}
        实现与官方 diffusers DDPMScheduler 保持一致

        参数:
            model_output: 模型输出（噪声、图像或速度）
            timestep: 当前时间步（int 或 torch.Tensor）
            sample: 当前噪声图像 x_t
            generator: 随机数生成器
            return_dict: 是否返回字典格式

        返回:
            prev_sample: 去噪后的图像 x_{t-1}
        """
        # 处理时间步：转换为标量
        if isinstance(timestep, torch.Tensor):
            t = timestep.item()
        else:
            t = int(timestep)

        prev_t = t - 1

        # 1. 计算 alphas 和 betas
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        # 当 prev_t < 0 时，使用 1.0（与官方调度器一致）
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t].to(sample.device)
        else:
            alpha_prod_t_prev = torch.tensor(
                1.0, device=sample.device, dtype=sample.dtype
            )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. 根据预测类型计算 pred_original_sample
        if self.prediction_type == "epsilon":
            # 模型预测噪声，使用公式 (15) 计算 x_0
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            # 模型直接预测去噪后的图像
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            # 模型预测速度
            pred_original_sample = (
                alpha_prod_t ** (0.5) * sample - beta_prod_t ** (0.5) * model_output
            )
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # 3. 裁剪或阈值处理（可选）
        if self.thresholding:
            # 动态阈值
            s = torch.quantile(
                torch.abs(pred_original_sample).reshape(
                    pred_original_sample.shape[0], -1
                ),
                self.dynamic_thresholding_ratio,
                dim=1,
            )
            s = torch.clamp(s, min=1.0).view(
                -1, *([1] * (pred_original_sample.ndim - 1))
            )
            pred_original_sample = pred_original_sample.clamp(-s, s) / s
        elif self.clip_sample:
            # 裁剪到 [-1, 1]
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        # 4. 计算系数（公式 (7)）
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = (
            current_alpha_t ** (0.5) * beta_prod_t_prev
        ) / beta_prod_t

        # 5. 计算 pred_prev_sample 的均值（公式 (7)）
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. 添加噪声（方差）
        variance = 0.0
        if t > 0:
            # 计算方差
            if self.variance_type == "fixed_small":
                variance = self.posterior_variance[t].to(sample.device)
            elif self.variance_type == "fixed_large":
                variance = self.betas[t].to(sample.device)
            elif self.variance_type == "learned":
                # 如果模型学习方差，这里需要调整
                variance = model_output[:, -1:]  # 假设最后通道是方差
            else:
                raise ValueError(f"Unknown variance_type: {self.variance_type}")

            # 添加随机噪声
            if generator is None:
                noise = torch.randn(
                    pred_prev_sample.shape,
                    device=pred_prev_sample.device,
                    dtype=pred_prev_sample.dtype,
                )
            else:
                noise = torch.randn(
                    pred_prev_sample.shape,
                    generator=generator,
                    device=pred_prev_sample.device,
                    dtype=pred_prev_sample.dtype,
                )

            pred_prev_sample = pred_prev_sample + (variance ** (0.5)) * noise

        if not return_dict:
            return pred_prev_sample

        return {
            "prev_sample": pred_prev_sample,
            "pred_original_sample": pred_original_sample,
        }

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        """缩放模型输入（DDPM不需要缩放，为了API兼容性保留）"""
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """
        设置推理时的时间步序列

        参数:
            num_inference_steps: 推理步数
            device: 设备
        """
        self.num_inference_steps = num_inference_steps

        # 创建步长序列（均匀采样）
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(
            0, self.num_train_timesteps, step_ratio, dtype=torch.long, device=device
        ).flip(0)

        self.timesteps = timesteps
        return timesteps

    def get_alpha_beta(self, t: torch.Tensor) -> tuple:
        """获取指定时间步的alpha和beta值"""
        t = t.long()
        alpha = self.alphas.to(t.device)[t]
        beta = self.betas.to(t.device)[t]
        return alpha, beta

    def visualize_schedule(self):
        """可视化调度器的参数变化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Beta序列
        axes[0, 0].plot(self.betas.numpy())
        axes[0, 0].set_title("Beta Schedule")
        axes[0, 0].set_xlabel("Timestep")
        axes[0, 0].set_ylabel("Beta")

        # 2. Alpha累积乘积
        axes[0, 1].plot(self.alphas_cumprod.numpy())
        axes[0, 1].set_title("Alpha Cumprod")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("ᾱ")

        # 3. 信噪比 (SNR = ᾱ / (1 - ᾱ))
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        axes[0, 2].plot(snr.numpy())
        axes[0, 2].set_title("Signal-to-Noise Ratio")
        axes[0, 2].set_xlabel("Timestep")
        axes[0, 2].set_ylabel("SNR")
        axes[0, 2].set_yscale("log")

        # 4. 后验方差
        axes[1, 0].plot(self.posterior_variance.numpy())
        axes[1, 0].set_title("Posterior Variance")
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("σ²")

        # 5. sqrt(alpha_cumprod)
        axes[1, 1].plot(self.sqrt_alphas_cumprod.numpy())
        axes[1, 1].set_title("sqrt(ᾱ)")
        axes[1, 1].set_xlabel("Timestep")

        # 6. sqrt(1 - alpha_cumprod)
        axes[1, 2].plot(self.sqrt_one_minus_alphas_cumprod.numpy())
        axes[1, 2].set_title("sqrt(1 - ᾱ)")
        axes[1, 2].set_xlabel("Timestep")

        plt.tight_layout()
        plt.show()

    def __len__(self):
        """返回总时间步数"""
        return self.num_train_timesteps


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 1. 创建调度器
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="fixed_small",
    )

    # 可视化参数
    print("调度器参数:")
    print(f"总时间步数: {len(scheduler)}")
    print(f"Beta范围: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
    print(
        f"Alpha累积乘积范围: [{scheduler.alphas_cumprod[0]:.6f}, {scheduler.alphas_cumprod[-1]:.6f}]"
    )

    # 2. 测试前向加噪过程
    print("\n测试前向加噪过程:")
    batch_size = 4
    image_size = 32
    channels = 3

    # 创建随机图像和噪声
    original_samples = torch.randn(batch_size, channels, image_size, image_size)
    noise = torch.randn_like(original_samples)

    # 随机时间步
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,))

    # 添加噪声
    noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
    print(f"原始图像形状: {original_samples.shape}")
    print(f"噪声图像形状: {noisy_samples.shape}")
    print(f"时间步: {timesteps.tolist()}")

    # 3. 测试反向去噪过程
    print("\n测试反向去噪过程:")

    # 设置推理步数
    scheduler.set_timesteps(num_inference_steps=50)
    print(f"推理时间步: {scheduler.timesteps[:10].tolist()}...")

    # 模拟模型输出（预测的噪声）
    model_output = torch.randn_like(noisy_samples)

    # 执行一步去噪
    result = scheduler.step(
        model_output=model_output,
        timestep=500,  # 中间时间步
        sample=noisy_samples[0:1],  # 只处理一个样本
        return_dict=True,
    )

    print(f"输入形状: {noisy_samples[0:1].shape}")
    print(f"输出形状: {result['prev_sample'].shape}")
    print(f"预测的原始样本形状: {result['pred_original_sample'].shape}")

    # 4. 可视化调度器参数
    scheduler.visualize_schedule()
