# Crowd Counting with MCNN Models

## Overview

This project implements a **crowd counting system** using the **Multi-Column Convolutional Neural Network (MCNN)** architecture. The system predicts the number of people in an uploaded image and visualizes their density distribution using a heatmap. It supports multiple pre-trained models for accurate predictions.

---

## Features

- **Accurate Crowd Counting**: Uses pre-trained MCNN models trained on the ShanghaiTech dataset.
- **Interactive Streamlit Interface**: Upload images, select models, and visualize results with ease.
- **Density Map Visualization**: Displays a heatmap showing density distribution in the image.
- **Pre-trained Models**: Fine-tuned on dense and sparse crowd datasets.

---

## Datasets Used

- **ShanghaiTech Dataset**:
    - **Part A**: Dense urban crowd scenes.
    - **Part B**: Sparse suburban scenes.
- **Preprocessing**:
    - Resized images and corresponding density maps.
    - Gaussian kernels were used to create density maps.

---

## Results

| **Model**                | **Dataset** | **Mean Absolute Error (MAE)** | **Mean Squared Error (MSE)** |
|--------------------------|-------------|-------------------------------|------------------------------|
| `Basic_Model.pth`        | Part A      | 197                           | 276                          |
| `Intermediate_Model.pth` | Part A      | 192                           | 269                          |
| `Advanced_Model.pth`     | Combined    | 177                           | 253                          |

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/KhuramMirza/crowd_counting_models.git
   cd crowd-counting-mcnn
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Download the ShanghaiTech Dataset from ShanghaiTech Dataset Link.
   Extract the dataset and place it in the data/ directory. The directory structure should look like this:

   ``` 
   data/
   ├── ShanghaiTech/
   │   ├── part_A/
   │   │   ├── train_data/
   │   │   ├── test_data/
   │   ├── part_B/
   │   │   ├── train_data/
   │   │   ├── test_data/
   
   ```


4**Prepare Pre-trained Models**:
   Place your pre-trained `.pth` files in the `models/` directory.

---

## Usage

### **Run the Application**

Start the Streamlit app:

```bash
streamlit run app.py
```

### **Predict Crowd Count**

1. Upload a crowd image via the interface.
2. Select a pre-trained model from the dropdown menu.
3. Click **Predict Headcount** to view the predicted count and density map.

---

## Project Structure

```
deploy_model/
├── app.py                   # Main Streamlit application
├── run_preprocess.py        # Preprocess ShanghaiTech Dataset
├── run_train.py             # Train Model
├── data/                    # Dataset 
│   ├── ShanghaiTech/
│   │   ├── part_A/
│   │   │   ├── train_data/
│   │   │   ├── test_data/
│   │   ├── part_B/
│   │   │   ├── train_data/
│   │   │   ├── test_data/
├── models/                  # Pre-trained models directory
│   ├── Basic_Model.pth
│   ├── Intermediate_Model.pth
│   └── Advanced_Model.pth
├── src/                     # Source files
│   ├── model.py             # MCNN model architecture
│   ├── preprocess.py        # Data preprocessing utilities
│   ├── dataset.py           # Dataset loader
│   └── train.py             # Training script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Dependencies

The project uses the following libraries:

- **PyTorch**: For training and inference.
- **Streamlit**: For building the user interface.
- **OpenCV**: For image preprocessing.
- **NumPy**: For numerical computations.
- **Matplotlib**: For density map visualization.

---

## Future Improvements

- Train the model on additional datasets for better generalization.
- Explore advanced architectures (e.g., CSRNet) for improved accuracy.
- Extend functionality to real-time video-based crowd counting.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
