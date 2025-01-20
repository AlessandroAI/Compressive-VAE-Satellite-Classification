# Data Directory

This branch contains datasets required for the project.

## Directory Structure
- `raw/`: Contains raw datasets downloaded or manually added.
  - `EuroSAT/`: Raw EuroSAT dataset.
  - `RSI-CB256/`: Raw RSI-CB256 dataset.
  - `PatterNet/`: Raw PatterNet dataset.

## Adding Data
1. **EuroSAT**:
   - Download the dataset from [EuroSAT GitHub](https://github.com/phelber/EuroSAT).
   - Place the unzipped folder in `data/raw/EuroSAT/`.

2. **RSI-CB256**:
   - Download from [RSI-CB256 Dataset Source](https://github.com/lehaifeng/RSI-CB).
   - Place the unzipped data in `data/raw/RSI-CB256/`.

3. **PatterNet**:
   - Download from [PatterNet Dataset Source](https://huggingface.co/datasets/blanchon/PatternNet).
   - Place the unzipped data in `data/raw/PatterNet/`.

## Usage
Once datasets are prepared, the scripts in the `src/` folder will handle preprocessing and loading. Update the `load_dataset` function in `utils.py` if new datasets are added.
