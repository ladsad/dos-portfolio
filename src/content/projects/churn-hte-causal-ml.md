# Churn Prediction with Heterogeneous Treatment Effects

A causal ML system that predicts customer churn **and** determines the best personalized intervention for each customer using causal inference.

## Key Innovation
Most churn systems just predict "who churns". This project uses **Causal Forests** to estimate **Conditional Average Treatment Effects (CATE)**, identifying customers who will specifically respond to an intervention, rather than wasting resources on lost causes or loyal customers.

## Methodology
1.  **Churn Prediction**: LightGBM model (AUC ~0.82) to identify at-risk customers.
2.  **Causal Effect Estimation**: Doubly Robust Estimation to isolate true intervention effects, removing selection bias.
3.  **Heterogeneity**: Causal Forests (econml) to estimate per-customer treatment effects.
4.  **Production**: FastAPI service for real-time recommendations.

## Results
-   **Average Treatment Effect**: -12% (Intervention reduces churn probability by 12pp).
-   **A/B Test Validation**: Targeted interventions on high-CATE segments yielded a **24% relative churn reduction** compared to random targeting.
-   **Business Impact**: Positive ROI (~2x) by optimizing intervention costs.

## Tech Stack
-   **Causal ML**: EconML, CausalML, DoWhy
-   **Machine Learning**: LightGBM, Scikit-Learn
-   **Production**: FastAPI, Docker
-   **Analysis**: Jupyter, Pandas, Matplotlib
