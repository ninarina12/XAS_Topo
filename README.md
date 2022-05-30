## Machine learning spectral indicators of topology

### Workflow
1. `Data assembly` directory: Scripts for querying, filtering, and processing spectral and label data.
2. `pca_kmeans.ipynb`: Notebook for conducting an exploratory analysis of the data, including principal component analysis and k-means clustering of XAS spectra.
3. `topology_classifier.ipynb`: Notebook for training a neural network classifier of band topology from XAS spectral inputs.

### Installation
1. Clone the repository:
    > `git clone https://github.com/ninarina12/XAS_Topo.git`

    > `cd XAS_Topo`

2. Create a virtual environment for the project:
    > `conda create -n xas_topo python=3.7.5`

    > `conda activate xas_topo`

3. Install packages:
    > `pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html`

    where `${TORCH}` and `${CUDA}` should be replaced by the specific CUDA version (e.g. `cpu`, `cu102`) and PyTorch version (e.g. `1.9.1`), respectively. For example:

    > `pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.9.1+cu102.html`


### References 
N. Andrejevic\*, J. Andrejevic\*, B. A. Bernevig, N. Regnault, C. H. Rycroft, and M. Li. *(\*=equal contributions) Machine learning spectral indicators of topology.* arXiv preprint arXiv:2003.00994 (2020).