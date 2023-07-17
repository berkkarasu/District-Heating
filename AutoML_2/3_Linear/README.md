# Summary of 3_Linear

[<< Go back](../README.md)


## Linear Regression (Linear)
- **n_jobs**: -1
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True

## Optimized metric
rmse

## Training time

1.6 seconds

### Metric details:
| Metric   |      Score |
|:---------|-----------:|
| MAE      |  2.19484   |
| MSE      | 10.7328    |
| RMSE     |  3.27609   |
| R2       |  0.896602  |
| MAPE     |  0.0861955 |



## Learning curves
![Learning curves](learning_curves.png)

## Coefficients
| feature                   |   Learner_1 |
|:--------------------------|------------:|
| Glazing_Area              |   1.71594   |
| Relative_Compactness      |   0.773571  |
| Overall_Height            |   0.697083  |
| Surface_Area              |   0.565035  |
| Roof_Area                 |   0.448041  |
| Cooling_Load              |   0.424676  |
| Wall_Area                 |   0.232472  |
| Glazing_Area_Distribution |   0.0798822 |
| Orientation               |  -0.0223126 |
| intercept                 |  -0.404854  |


## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



## SHAP Importance
![SHAP Importance](shap_importance.png)

## SHAP Dependence plots

### Dependence (Fold 1)
![SHAP Dependence from Fold 1](learner_fold_0_shap_dependence.png)

## SHAP Decision plots

### Top-10 Worst decisions (Fold 1)
![SHAP worst decisions from fold 1](learner_fold_0_shap_worst_decisions.png)
### Top-10 Best decisions (Fold 1)
![SHAP best decisions from fold 1](learner_fold_0_shap_best_decisions.png)

[<< Go back](../README.md)
