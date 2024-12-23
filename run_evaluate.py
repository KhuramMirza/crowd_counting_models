from src.evaluate import evaluate

# Define paths to model and datasets
model_path = "mcnn_finetuned_v2.pth"  # Path to your trained model
data_dir_A = "data/processed/part_A/test"  # Path to Part_A test data
# data_dir_B = "data/processed/part_B/test"  # Path to Part_B test data (optional)

# Run evaluation
evaluate(
    model_path=model_path,
    data_dir_A=data_dir_A,
    # data_dir_B=data_dir_B,  # Set to None if not using Part_B
    batch_size=4
)
