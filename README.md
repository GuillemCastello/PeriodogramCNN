# PeriodogramCNN

A simple Python project to process solar telescope data, look at how image brightness varies over time, and train a small neural network to find noise characteristics in frequency plots (periodograms).

## 📋 Features

- **Download data**: Get GONG H‑alpha FITS files using a provided script.
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

## 📥 Downloading Data

A script to grab example FITS files is in the `data/` folder. To download sample data:
```bash
bash data/download_gong_halpha_fits_ata.sh
```
This will save files under `data/` by date.

## 🔄 Data Processing

1. Open `data_processing/preprocess_data.py` and set:
   - `day` to the date folder name (e.g., `20140125`).
   - `directory_of_data` to where your `.fits.fz` files live (e.g., `/home/user/20140125/`).
2. Run preprocessing for one day:
   ```bash
   python data_processing/preprocess_data.py <num_processes>
   ```
   Processed files go into an `updated/` subfolder.

3. To batch-process many days, use:
   ```bash
   bash preprocess_multiple_days.sh
   ```

## 🗺️ Mapping

Inside `data_processing/mapping/` you'll find scripts to turn FITS images into solar disk maps:
- `fits2map.py`: read FITS data into arrays.
- `make_map.py`: build and save map files.
- `plot_map.py`: display map images.
- Additional utilities for coordinate transforms and plotting.

## 📊 Analysis Notebooks

- **psd_analysis.ipynb**: Shows how to compute and fit periodograms (power vs. frequency plots).
- **plotting.ipynb**: Examples of plotting cleaned data and model fits.

Open these with Jupyter:
```bash
jupyter notebook psd_analysis.ipynb
```

## 🤖 Training the CNN

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

