# train.py
import pytorch_lightning as pl
from config import config
from data_module import MNISTDataModule
from diffusion_module import DiffusionExperiment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def main():
    # 1. 初始化数据模块
    dm = MNISTDataModule()

    # 2. 初始化模型
    model = DiffusionExperiment()

    # 3. 设置回调函数
    # 模型检查点：保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.save_dir,
        filename="best-checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # 早停：防止过拟合
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    # 4. 初始化训练器
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",  # 自动选择GPU/CPU
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        default_root_dir=config.save_dir,
        prediction_type=config.prediction_type,
    )

    # 5. 开始训练！
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
