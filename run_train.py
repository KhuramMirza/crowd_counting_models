from src.train import train


train(
    data_dir_A="data/processed/part_A/train",
    data_dir_B="data/processed/part_B/train",
    model_path="mcnn_combined.pth",
    epochs=20,
    batch_size=8,
    learning_rate=5e-5,
    validation_split=0.2
)
