# Model Explainability for Fraud Detection

## Task 3: Model Explainability

Model explainability is a critical aspect of machine learning, especially in sensitive applications like fraud detection. This repository demonstrates the use of SHAP (Shapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to interpret machine learning models for fraud detection.

In this task, we use SHAP and LIME to interpret and explain the models we have built for fraud detection, helping to understand the predictions and trust the results.

## SHAP (Shapley Additive exPlanations)

SHAP is a unified measure of feature importance that helps explain the contribution of each feature to the prediction of a model.
The SHAP explanation framework provides several ways to visualize the impact of features on model predictions:
1. SHAP Summary Plot
The Summary Plot provides an overview of the most important features across the dataset, helping you identify which features contribute the most to the model's predictions.
2. SHAP Force Plot
The Force Plot visualizes the contribution of features for a single prediction, allowing you to see how individual features push the model output in one direction or another.
3. SHAP Dependence Plot
The Dependence Plot shows the relationship between a single feature and the model output, providing insight into how a feature's value affects the prediction.
### Installing SHAP

To install SHAP, use the following command:

```bash
pip install shap
```
## LIME (Local Interpretable Model-agnostic Explanations)
LIME is used to explain individual predictions by approximating the model locally with an interpretable model. It provides insights into which features contribute most to the model's prediction for a single instance.

Installing LIME
To install LIME, use the following command:

```bash
pip install lime
```

LIME creates feature importance plots by generating an interpretable model locally around a specific prediction. Here's how you can use LIME:

1. LIME Feature Importance Plot
The Feature Importance Plot shows the most influential features for a specific prediction, helping to understand why a particular prediction was made.
## Conclusion
Model explainability using SHAP and LIME helps in building trust in machine learning models, especially when deployed in critical applications like fraud detection. By understanding which features contribute to predictions, users can gain more insights into model behavior and improve its performance if necessary.

### Key Features:
- **SHAP**: Explains model predictions by calculating feature importance across the entire dataset.
- **LIME**: Explains individual predictions using locally interpretable models.
