## Data assembly

### Workflow
1. `filter_icsd.py`: Query the Materials Project database [1] by ICSD identifier of materials in the topological materials database [2] and collect matching mp-ids
2. `query_xas.py`: Query the Materials Project database by mp-id for all available XAS spectra [3]
3. `filter_xas.ipynb`: Filter out empty or non-numeric queries of XAS spectra

### Notes
* Files ending in `_manual` apply the same workflow to a manually-digitized database of Ref. 4. An initial processing step is handled by `process_icsd_manual.ipynb`.
* Identify known Weyl semimetals present in the dataset with `filter_weyl.ipynb`.

### References 
[1] A. Jain\*, S.P. Ong\*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson *(\*=equal contributions). The Materials Project: A materials genome approach to accelerating materials innovation.* APL Materials, 2013, 1(1), 011002.
[2] M. G. Vergniory, B. J. Wieder, L. Elcoro, S. S. Parkin, C. Felser, B. A. Bernevig, N. Regnault, *All topological bands of all stoichiometric materials.* arXiv preprint, 2021, arXiv:2105.09954.
[3] K. Mathew, C. Zheng, D. Winston, C. Chen, A. Dozier, J. J. Rehr, S. P. Ong, K. A. Persson. *High-throughput computational X-ray absorption spectroscopy.* Scientific Data, 2018, 5.
[4] M. G. Vergniory, L. Elcoro, C. Felser, N. Regnault, B. A. Bernevig, Z. Wang. *A complete catalogue of high-quality topological materials.* Nature, 2019, 566(7745), 480-485.