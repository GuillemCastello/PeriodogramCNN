# PeriodogramCNN

A simple Python project to process solar telescope data, look at how image brightness varies over time, and train a small neural network to find noise characteristics in frequency plots (periodograms).

## 📋 Features

- **Preprocess images**: Remove limb darkening (the Sun appears darker at its edges) and filter out bad frames.
- **Make maps**: Turn FITS files into 2D maps of the solar disk and plot them.
- **Analyze periodograms**: Compute power versus frequency (periodogram) and fit a simple noise model.
- **Train a CNN**: A 1D convolutional neural network learns to predict noise parameters from periodograms.

## ⚙️ Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/PeriodogramCNN.git
   cd PeriodogramCNN
   ```
2. **Install dependencies** (requires Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```

## 🔄 Data Processing
1. Run preprocessing for one day:
   ```bash
   bash preprocess_multiple_days.sh
   ```
   Change day and ncores as needed, your data should be isnide the data folder

## 📊 Analysis Notebooks

- **psd_analysis.ipynb**: Shows how to compute and fit periodograms (power vs. frequency plots).
- **plotting.ipynb**: Examples of plotting cleaned data and model fits.

Open these with Jupyter:
```bash
jupyter notebook psd_analysis.ipynb
```

## 🤖 CNN

1. The CNN weights presented in Castelló et al. (2025) for both models are present CNN/*/*Weights.h5
2. Load your models using tensorflow with the usual routines and you are ready to go to eprform inference.

## 🤖 Training the CNN (if needed)

1. Go to the `CNN/` folder:
   ```bash
   cd CNN
   ```
2. Run the training script:
   ```bash
   python train.py
   ```
   - This reads periodogram data and known noise parameters from HDF5 files.
   - Trains a simple 1D CNN to predict the parameters.
   - Saves best weights under `CNN/BestFit/`.

## 🗂️ Project Structure

```
PeriodogramCNN/
├── CNN/                   # Code and saved models for CNN training
├── data/                  # Scripts to download example FITS data
├── data_processing/       # Preprocessing scripts and mapping utilities
│   └── mapping/           # FITS-to-map converters and plotting tools
├── psd_analysis.ipynb     # Notebook for periodogram analysis
├── plotting.ipynb         # Notebook for example plots
├── preprocess_multiple_days.sh  # Batch preprocessing shell script
├── requirements.txt       # Python package list
└── README.md              # This file
```

## 🤝 Contributing

Contributions and feedback are welcome! Feel free to open an issue or submit a pull request.

