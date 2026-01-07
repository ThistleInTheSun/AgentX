# config.py
class Config:
    # 数据相关
    dataset_name = "MNIST"
    image_size = 32  # 将MNIST上采样到32x32，适配更多模型
    batch_size = 128
    
    # 扩散过程相关
    num_train_timesteps = 1000  # 扩散总步数
    beta_schedule = "linear"    # 噪声调度类型
    
    # 模型相关
    # 使用Diffusers库内置的UNet2D，后续可轻松替换
    model_config = {
        "sample_size": image_size,        # 图像尺寸
        "in_channels": 1,                 # 输入通道（MNIST为灰度）
        "out_channels": 1,                # 输出通道
        "layers_per_block": 2,            # 每层块数（控制深度）
        "block_out_channels": (32, 64, 128),  # 各层通道数（控制宽度）
        "down_block_types": (
            "DownBlock2D",    # 下采样块
            "AttnDownBlock2D", # 带注意力的下采样块
            "AttnDownBlock2D",
        ),
        "up_block_types": (
            "AttnUpBlock2D",  # 带注意力的上采样块
            "AttnUpBlock2D",
            "UpBlock2D",      # 上采样块
        ),
    }
    
    # 训练相关
    lr = 1e-4
    epochs = 30
    save_dir = "./results"
    
config = Config()