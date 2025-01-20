#Optimizing Satellite Image Analysis: Leveraging Variational Autoencoders for Direct Classification

Official repository for the paper "Optimizing Satellite Image Analysis: Leveraging Variational Autoencoders Latent Representations for Direct Integration"

##Overview

This repository contains the implementation of the methodologies described in our publication:
"Optimizing Satellite Image Analysis: Leveraging Variational Autoencoders for Direct Classification."

In this study, we propose using latent representations from Variational Autoencoders (VAEs) directly for downstream machine learning tasks like classification, without requiring explicit reconstruction. Our experiments demonstrate significant improvements in classification performance while maintaining efficient compression and reconstruction quality.

##Features

Implementation of neural compression models using advanced Variational Autoencoder architectures.

Fine-tuning workflows to optimize latent representations for classification tasks.

Comprehensive evaluation metrics, including Rate Distortion Accuracy Index (RDAI).

Support for popular satellite image datasets: EuroSAT, RSI-CB256, and PatternNet.

##Repository Structure

Satellite_Image_Analysis_with_VAE/
├── README.md              # Overview of the project
├── LICENSE                # Licensing information
├── requirements.txt       # Python dependencies
├── notebooks/             # Jupyter notebooks for experiments
│   ├── Compress_AI_models_1.ipynb
│   ├── Compress_AI_models_2_FT.ipynb
│   ├── Compress_AI_models.ipynb
├── src/                   # Python scripts
│   ├── vae.py             # Implementation of VAE architectures
│   ├── fine_tuning.py     # Fine-tuning and training logic
│   ├── utils.py           # Helper functions for data processing
│   ├── evaluation.py      # Metrics and visualization tools
├── data/                  # Placeholder for datasets
│   ├── EuroSAT/           # Processed dataset
│   ├── raw/               # Raw dataset (unprocessed)
├── results/               # Directory for results
│   ├── tSNE_plots/        # t-SNE visualizations
│   ├── metrics.json       # Summary of metrics
│   ├── reconstructed_images/
├── docs/                  # Documentation
│   ├── publication.md     # Link to the paper and context
│   ├── usage.md           # Instructions for usage
│   ├── architecture.png   # Model architecture diagrams
└── .gitignore             # Files to ignore in the repository

##Installation

Clone the repository:

git clone https://github.com/your_username/Satellite_Image_Analysis_with_VAE.git
cd Satellite_Image_Analysis_with_VAE

Install dependencies:

pip install -r requirements.txt

##Usage

Running Notebooks

Open the Jupyter notebooks in the notebooks/ folder to:

Train neural compression models.

Evaluate classification tasks on latent representations.

Visualize t-SNE plots of latent spaces.

##Running Scripts

Prepare the dataset in the data/raw/ directory.

Train and evaluate models using the scripts in the src/ folder:

python src/vae.py --dataset EuroSAT --epochs 50
python src/fine_tuning.py --vae_model cheng2020_attn --classifier Transformer

##Datasets

EuroSAT: Sentinel-2 satellite imagery dataset.

RSI-CB256: Land-use classification dataset.

PatternNet: Urban and suburban satellite image dataset.

##Results

###Metrics

BPP (Bits Per Pixel)

PSNR (Peak Signal-to-Noise Ratio)

F1 Score

Rate Distortion Accuracy Index (RDAI)

###Visualizations

t-SNE plots of latent spaces highlight clustering and separability.

Example reconstruction images are provided in the results/reconstructed_images/ folder.

##Citation

If you use this repository, please cite our paper:

@article{YourArticle,
  title={Optimizing Satellite Image Analysis: Leveraging Variational Autoencoders for Direct Classification},
  author={Alessandro Giuliano, S. Andrew Gadsden, John Yawney},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}

##License

This project is licensed under the MIT License. See the LICENSE file for details.

##Acknowledgments

CompressAI Library for neural compression models.

Support from McMaster University.

For more information, contact giuliana@mcmaster.ca.

