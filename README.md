# Clustering Algorithms - Machine Learning Course Assignment

This repository contains comprehensive Google Colab notebooks demonstrating various clustering algorithms and techniques.

## Notebooks Overview

| # | Notebook | Description | Key Algorithms/Libraries |
|---|----------|-------------|-------------------------|
| a | [K-Means from Scratch](a_kmeans_from_scratch.ipynb) | Implementation of K-Means clustering from scratch | K-Means, K-Means++ initialization |
| b | [Hierarchical Clustering](b_hierarchical_clustering.ipynb) | Agglomerative clustering with dendrograms | Ward, Complete, Single, Average linkage |
| c | [Gaussian Mixture Models](c_gaussian_mixture_models.ipynb) | GMM for soft clustering | sklearn GaussianMixture, BIC/AIC |
| d | [DBSCAN with PyCaret](d_dbscan_pycaret.ipynb) | Density-based clustering | PyCaret, DBSCAN |
| e | [Anomaly Detection (PyOD)](e_anomaly_detection_pyod.ipynb) | Outlier detection techniques | PyOD, IForest, LOF, OCSVM |
| f | [Time Series Clustering](f_timeseries_clustering.ipynb) | Clustering temporal data | tslearn, DTW, tsfresh |
| g | [Document Clustering (LLM)](g_document_clustering_llm.ipynb) | Text clustering with embeddings | Sentence-Transformers, UMAP |
| h | [Image Clustering](h_image_clustering_imagebind.ipynb) | Visual clustering with CLIP | CLIP, torchvision |
| i | [Audio Clustering](i_audio_clustering_embeddings.ipynb) | Audio feature clustering | librosa, Wav2Vec2 |

## Clustering Quality Metrics Used

All notebooks include proper evaluation using:
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Adjusted Rand Index (ARI)**: Agreement with ground truth (0 to 1, higher is better)
- **Normalized Mutual Information (NMI)**: Information overlap measure
- **Calinski-Harabasz Index**: Ratio of between/within cluster variance (higher is better)
- **Davies-Bouldin Index**: Average cluster similarity (lower is better)

## How to Use

1. Click on any notebook link above
2. Open in Google Colab: `File > Open in Colab` or use the Colab badge
3. Run all cells sequentially

Or click the badges below to open directly in Colab:

| Notebook | Open in Colab |
|----------|---------------|
| K-Means from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/a_kmeans_from_scratch.ipynb) |
| Hierarchical Clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/b_hierarchical_clustering.ipynb) |
| GMM Clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/c_gaussian_mixture_models.ipynb) |
| DBSCAN with PyCaret | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/d_dbscan_pycaret.ipynb) |
| Anomaly Detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/e_anomaly_detection_pyod.ipynb) |
| Time Series Clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/f_timeseries_clustering.ipynb) |
| Document Clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/g_document_clustering_llm.ipynb) |
| Image Clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/h_image_clustering_imagebind.ipynb) |
| Audio Clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/clustering-algorithms/blob/main/i_audio_clustering_embeddings.ipynb) |

## Requirements

All required packages are installed within each notebook. Key dependencies include:
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- pycaret
- pyod
- tslearn, tsfresh
- sentence-transformers
- torch, torchvision, torchaudio
- transformers
- librosa
- umap-learn, hdbscan

## References

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [PyCaret Clustering](https://pycaret.org/)
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [tslearn](https://tslearn.readthedocs.io/)

## Author

Created for Machine Learning Course Assignment
