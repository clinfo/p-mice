# p-mICE
Projected-multivariate Individual Conditional Expectation (p-mICE) is an XAI approach that extends ICE.  
p-mICE aims to visualize changes in predicted values with the perturbation of explanatory variables while accounting for multivariate dependence.  

## Requirements
CentOS Linux release 8.1 (Confirmed)  

### Create conda environment
```bash
conda config --add channels conda-forge
conda create --name p_mice python=3.7.3
conda activate p_mice
conda install pandas=0.25.3
conda install numpy=1.16.0
conda install xgboost=0.82
conda install scikit-learn=0.21.2
conda install matplotlib=3.1.2
conda install seaborn=0.11.2
conda install jupyter=1.0.0
```

## Tutorials
In brief, `Tutorial.ipynb` can be run on the Jupiter Notebook.  

### Import
```python
import p_mice
```

### p-mICE settings
```python
pdc_params = p_mice.PdcParams(*model, *ar_feature_names, *intervention_variables, threshold=0.5, roughness=0.05, cont_index=None)
```

- __\*model__: p-mICE can be applied to any binary classification ML model with `predict_proba` method.  
- __\*ar_feature_names__: 1d-array [str]. Feature names used in ML model. The array length should be the same as the number of explanatory variables in the model.  
- __\*intervention_variables__: List [str]. Two intervention variables corresponding to the x- and y-axes of the plot.  
- __threshold__: [float]. Threshold value of the prediction probability for binary classification in the prediction model (default=0.5).  
- __roughness__: [float] value in the range of [0, 1]. The smaller the roughness, the more finely the search is performed (default=0.05).  
- __cont_index__: 1d-array [bool] to indicate continuous features. If None, all features are regarded as continuous variables (default=None).  

### p-mICE calculation for the individual record
```python
pdc_params = p_mice.calc_p_mice(*pdc_params, *record, *ar_ref_X, ar_ref_y)
```

- __\*pdc_params__: p-mICE setting.  
- __\*record__: 2d-array. Single record to explain.  
- __\*ar_ref_X__: 2d-array. Reference data to search neighbor records in p-mICE.  
- __same_label__: [bool]. If True, reference data is restricted to records that have the same label as original predicted value (default=True).  
- __ar_ref_y__: 1d-array [int]. Labels for ar_ref_X. If same_label is True, this option is required.  
- __ar_p_min__: 1d-array [float]. Lower limits of variation for intervention variables. If None, lower limits in the ar_ref_X would be set (default=None).
- __ar_p_max__: 1d-array [float]. Upper limits of variation for intervention variables. If None, upper limits in the ar_ref_X would be set (default=None).
- __n_neighbors__: [int]. The number of neighbor records used in p-mICE (default=4).
- __mean_seight__: [str], choices=['simple', 'euclidean1', 'euclidean2', 'exp_euclidean1', 'exp_euclidean2']. The weights of neighbor records in p-mICE (default='exp_euclidean1').

### Plot of p-mICE results
```python
p_mice.plot_phase_diagrams(pdc_params)
```
- __\*pdc_params__: Calculated p-mICE object.  

## License
This edition of p-mICE is for evaluation, learning, and non-profit academic research purposes only, and a license is needed for any other uses. Please send requests on license or questions to nakamura.kazuki.88m[at]st.kyoto-u.ac.jp