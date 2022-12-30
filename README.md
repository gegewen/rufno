README.md

This repository contains the code and figures associated with the paper:

Chu, A.K.; Benson, S.M.; Wen, G. Deep-Learning-Based Flow Prediction for CO2 Storage in Shale–Sandstone Formations. Energies 2023, 16, 246. https://doi.org/10.3390/en16010246

------------------------------
Model:

Run_FNORUNet3_dP_4layer.py: trains dP model. Can pass arguments specifying parameters such as the training/validation data set size, error type, learning rate, modes, etc.

Run_FNORUNet3_SG_5layer.py: trains SG model.

Run_FNORUNet4_dP_4layer_0rerr.py: trains dP model, with a loss function excluding the r-error.

Run_FNORUNet4_SG_5layer_0rerr.py: trains SG model, with a loss function excluding the r-error.

FNORUNet_4layer_model.py: model architecture for RU-FNO with 4 ResNet layers.

FNORUNet_5layer_model.py: model architecture for RU-FNO with 5 ResNet layers.

------------------------------
Analysis:

analysis.ipynb: plots for analysis of shale case studies.

calculateErr.ipynb: calculate R2 scores and mean errors of models.

dataExample.ipynb: plots for examples from training data.

dataGenerationExample.ipynb: plots illustrating data generation methodology.

plotResults.ipynb: plots for model prediction results.

R2plots.ipynb: R2 histograms and scatter plots (Fig 2)

R2training.ipynb: plots of R2 score over training process (Fig 2)

sleipnerSim.ipynb: model prediction for Sleipner-like reservoir (Fig 11)

speedup.ipynb: calculation of model speedup (Table 3)

The .npy data and PyTorch model files referenced in the code are available upon request (not included here because they exceed the Github 100MB file size limit).

------------------------------
Figures:

.png files for figures are located in the Figures directory.
