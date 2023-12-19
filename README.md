# boulder-regression

This is code from a project at Østfold University College, where we used data from the Kilter Board app to predict the difficulty of a boulder problem.

The first part (p1-traditional folder) is a traditional machine learning approach, where I used random forest and xgboost.

The second part of the project (p2-deep) is a deep learning approach, where I used a fully connected neural network, and lots of CNNs (and a convolutional vision transformer) to do the same.

This was a group project, but the code I publish here was written by myself, I did not include code from the other group members (and no preprocessing code or data).

## p1-traditional Results

Random forest 5-fold cross-validation (baseline):
MAE: 2.19
R2: 0.49
MSE: 8.25

XGBoost, 5-fold cross-validation:
MAE: 1.93
R2: 0.58
MSE: 6.89

## p2-deep Results

More data was obtained for this part of the project, so the results are not directly comparable to the traditional approach.

Fully connected neural network (baseline):
MAE: 1.58
R2: 0.68
MSE: 4.56

### CNNs

| Network                          | Validation MSE | Validation MAE | Validation R2 | Train MSE | Train MAE | Train R2 | Tuning | Overfitting MSE | Overfitting MAE | Overfitting R2 | Overfitting |
| -------------------------------- | -------------- | -------------- | ------------- | --------- | --------- | -------- | ------ | --------------- | --------------- | -------------- | ----------- |
| RegNet y400mf                    | 4.3423         | 1.5915         | 0.724         | 0.5123    | 0.6801    | 0.9673   | 1      | 88%             | 57%             | 25%            | 0.57        |
| RegNet y32gf                     | 3.7925         | 1.4514         | 0.7621        | 0.9187    | 0.6527    | 0.9425   | 0      | 76%             | 55%             | 19%            | 0.50        |
| ConvNeXt Large                   | 5.0713         | 1.6981         | 0.6819        | 3.1574    | 1.3678    | 0.8026   | 0      | 38%             | 19%             | 15%            | 0.24        |
| Convolutional Vision Transformer | 6.6148         | 2.0248         | 0.5849        | 7.1195    | 1.6839    | 0.7173   | 0      | \-8%            | 17%             | 18%            | 0.09        |
| ShuffleNet v2 x1                 | 3.4862         | 2.5533         | 1.3986        | 1.0967    | 0.7817    | 0.869    | 0      | 69%             | 69%             | \-61%          | 0.26        |
| ShuffleNet v2 x2                 | 3.7472         | 1.4732         | 0.7678        | 2.5498    | 1.1552    | 0.8597   | 0      | 32%             | 22%             | 11%            | 0.21        |
| EfficientNet b0                  | 3.3821         | 1.3788         | 0.7891        | 2.5162    | 1.1221    | 0.862    | 0      | 26%             | 19%             | 8%             | 0.18        |
| EfficientNet v2s                 | 4.4058         | 1.5884         | 0.7222        | 0.6796    | 1.0611    | 0.8673   | 0      | 85%             | 33%             | 17%            | 0.45        |
| MnasNet 0 75                     | 5.9781         | 1.8987         | 0.6237        | 1.2736    | 0.993     | 0.8957   | 0      | 79%             | 48%             | 30%            | 0.52        |
| MobileNet v2                     | 6.1629         | 1.9131         | 0.6143        | 1.367     | 0.8081    | 0.9206   | 0      | 78%             | 58%             | 33%            | 0.56        |

---

## Main Takeaways

- More data leads to better results (also when 30k+ examples are available already)
- Finding the right network architecture for the problem is more effective than adding more parameters
- When training on a high performance cluster (HPC), make sure data is not loaded from a network disk :)
