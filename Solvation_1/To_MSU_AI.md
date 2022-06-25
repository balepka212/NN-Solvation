## About the work
This project is intended to compare different vectorization approaches rather than to create a new model for Solvation
Energy prediction. Thus the results mainly compared between each other and not much attention is drawn to comparing with
other similar models.
## Reproducibility
In this project 63 (8•8-1) ResNet based models, 63 (8•8-1) LinNet based models and 315 (5•(8•8-1)) KRR estimators were 
trained on MNSol dataset. To obtain the models one might use Jupyter Notebooks ([KRR](Examples/Example_KRR.ipynb), 
 [LinNet](Examples/Example_Lin.ipynb), [ResNet](Examples/Example_Res.ipynb)), find desired training file in
[Trainig files](Training_files) for LinNet or ResNet or edit [KRR file](Training_files/000_Consequent_KRR.py) for KRR. 
However the training process takes quite long time. Thus, I would consider choosing the model you are interested in and
manually train it. 

Another part of the work is devoted to testing best models on foreign datasets and investigating 
feature permutation importance, which could be reproduced in 
[Jupyter Notebook](Examples/Foreign_Datasets_Feature_Permutations.ipynb)

## Units
As this work considers Solvation Free Energy, most of the values are in kcal/mol. Actual data passed to models however 
was normilized to have mean, std = 0, 1 on MNSol datasetinstead of -5.1549, 2.8032. Generally the error function was MSE
but for easier comparison the values on the graphs were presented as RMS and scaled to kcal/mol.

## Trained models
Trained models are stored locally and available on request due to their large size. Trained KRR models are available
with [google drive link](https://drive.google.com/drive/folders/1SSmV2efZHku2CqQUVswASER0lo6xiW7Z?usp=sharing) to be put
in [Run_results](Run_results) folder
However, chosen models are put into [data folder](Examples/data) and they are downloaded in mentioned above
[Jupyter Notebook](Examples/Foreign_Datasets_Feature_Permutations.ipynb).  

