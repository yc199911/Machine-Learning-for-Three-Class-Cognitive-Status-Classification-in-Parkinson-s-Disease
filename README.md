# Machine Learning for Three-Class Cognitive Status Classification in Parkinson's Disease

## Overview

This repository contains the implementation of a two-stage machine learning framework for classifying cognitive status in Parkinson's Disease (PD) patients. The model distinguishes among three cognitive states: PD with normal cognition (PD-NC), PD with mild cognitive impairment (PD-MCI), and PD dementia (PDD).

## Key Features

- **Two-stage classification framework**: Stage 1 (PD-NC vs. non-PD-NC) and Stage 2 (PD-MCI vs. PDD)
- **Explainable AI**: Uses SHAP (SHapley Additive exPlanations) for model interpretability
- **Ensemble approach**: Combines XGBoost and Multilayer Perceptron (MLP) classifiers
- **Class imbalance handling**: SMOTE-Tomek method for balanced learning
- **Subitem-level features**: Fine-grained clinical and neuropsychological assessments
- **Clinical feasibility**: Uses only routine clinical scales without neuroimaging

## Dataset

This study uses data from the Parkinson's Progression Markers Initiative (PPMI):
- **Total participants**: 1,439 individuals with PD
- **PD-NC**: 1,030 participants
- **PD-MCI**: 330 participants  
- **PDD**: 79 participants

### Data Access
- Data available from [PPMI database](https://www.ppmi-info.org/)
- Requires registration and approval by PPMI Data Access Committee
- Institutional ethics approval required

### Features Used
- **UPDRS** (NP1-NP4): Unified Parkinson's Disease Rating Scale subscales
- **MoCA**: Montreal Cognitive Assessment (total and subdomains)
- **VF_LT**: Verbal Fluency Letter Task
- **JLO**: Judgment of Line Orientation
- **ESS**: Epworth Sleepiness Scale
- **GDS**: Geriatric Depression Scale
- **Clock drawing**: Six individual items

## Requirements

```
Python >= 3.9
scikit-learn >= 1.2
XGBoost >= 1.7
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
shap >= 0.40.0
imbalanced-learn >= 0.8.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yc199911/Machine-Learning-for-Three-Class-Cognitive-Status-Classification-in-Parkinson-s-Disease.git
cd Machine-Learning-for-Three-Class-Cognitive-Status-Classification-in-Parkinson-s-Disease
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```python
from src.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load and preprocess PPMI data
X_train, X_test, y_train, y_test = preprocessor.load_and_split_data('path/to/ppmi_data.csv')
```

### Two-Stage Model Training
```python
from src.two_stage_model import TwoStageClassifier

# Initialize two-stage classifier
model = TwoStageClassifier(
    stage1_model='xgboost',
    stage2_model='mlp',
    use_smote_tomek=True,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### SHAP Analysis
```python
from src.interpretability import SHAPAnalyzer

# Initialize SHAP analyzer
shap_analyzer = SHAPAnalyzer(model)

# Generate SHAP explanations
shap_analyzer.explain_predictions(X_test)
shap_analyzer.plot_feature_importance()
```

## Model Architecture

### Stage 1: PD-NC vs. Non-PD-NC
- **Algorithm**: XGBoost
- **Features**: Top 10 SHAP-selected features
- **Balancing**: SMOTE-Tomek resampling
- **Validation**: 5-fold stratified cross-validation

### Stage 2: PD-MCI vs. PDD  
- **Algorithm**: Multilayer Perceptron (MLP)
- **Architecture**: Two hidden layers (100, 50 neurons)
- **Features**: Top 10 features + interaction terms
- **Activation**: ReLU
- **Optimizer**: Adam

## Performance Results

### Overall Three-Class Classification
- **Accuracy**: 0.92
- **Macro F1-score**: 0.79
- **Weighted F1-score**: 0.92

### Per-Class Performance
| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| PD-NC  | 0.94      | 1.00   | 0.97     |
| PD-MCI | 0.96      | 0.71   | 0.81     |
| PDD    | 0.50      | 0.71   | 0.59     |

### ROC-AUC Scores
- **PD-NC**: 0.85
- **PD-MCI**: 0.85  
- **PDD**: 0.84

## Key Findings

1. **Superior performance** compared to traditional screening methods (37% improvement over MoCA-only)
2. **Subitem-level features** significantly outperform total scores (107% accuracy increase)
3. **Clinical interpretability** through SHAP analysis reveals key predictive patterns
4. **Balanced detection** across all cognitive subgroups, including minority PDD class



## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{chen2024subitem,
  title={Subitem-Level Multi-Scale Assessment and Machine Learning for Three-Class Cognitive Status Classification in Parkinson's Disease},
  author={Chen, Ying-Che and Yu, Rwei-Ling and Hsieh, Sun-Yuan},
  journal={[Journal Name]},
  year={2024},
  note={In review}
}
```

## Contributing

We welcome contributions to improve this framework. Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Corresponding Authors:**
- Sun-Yuan Hsieh: hsiehsy@mail.ncku.edu.tw
- Rwei-Ling Yu: lingyu@mail.ncku.edu.tw

**First Author:**
- Ying-Che Chen: q56121036@gs.ncku.edu.tw

**Institution:** National Cheng Kung University, Taiwan

## Acknowledgments

- Parkinson's Progression Markers Initiative (PPMI) for providing the dataset
- Movement Disorder Society for cognitive assessment guidelines
- All PPMI participants and research teams

## Disclaimer

This code is provided for research purposes only. The model has not been clinically validated for diagnostic use. Please consult healthcare professionals for medical decisions.
