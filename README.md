# AI-FAPS: Self-, Semi-, and Combined Deep Learning Pipeline

Comparative pipeline for industrial visual inspection with three tracks:

- Self-supervised learning
- Semi-supervised learning
- Combined logic (self-supervised backbone + semi-supervised training flow)

## Project Structure

```text
ai-faps-self-semi-combined-dl-pipeline-industrial-inspection/
├── CombinationLogicFinal/
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── datasets.py
│   ├── hyperparameter_optimization/
│   │   └── hpo.py
│   ├── test/
│   │   ├── __init__.py
│   │   └── inference_combination.py
│   ├── train/
│   │   ├── __init__.py
│   │   └── train_combination.py
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py
│       └── manualseedsutils.py
├── Self-Supervised-Learning/
│   ├── data/
│   │   ├── __init__.py
│   │   └── Dataset.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── make_model.py
│   │   └── train_validation_test.py
│   ├── SSL_Pretrain/
│   │   └── simclr.py
│   ├── Test/
│   │   └── Test.py
│   ├── Training/
│   │   ├── Hyperparameter_optimization.py
│   │   └── Train_supervised_downstream.py
│   └── utils/
│       ├── __init__.py
│       └── Utils.py
├── Semi-Supervised-Learning/
│   ├── main.py
│   ├── configfiles/
│   │   ├── configfixmatchdino10.yaml
│   │   ├── configfixmatchdino25.yaml
│   │   ├── configfixmatchdino50.yaml
│   │   ├── configfixmatchdino100.yaml
│   │   ├── configfixmatchefficienet10.yaml
│   │   ├── configfixmatchefficienet25.yaml
│   │   ├── configfixmatchefficienet50.yaml
│   │   └── configfixmatchefficienet100.yaml
│   ├── dataset/
│   ├── models/
│   ├── testing/
│   ├── train/
│   └── utils/
├── .gitignore
└── README.md
```

## Quick Start

### 1) Clone and enter project

```bash
git clone <your-repository-url>
cd ai-faps-self-semi-combined-dl-pipeline-industrial-inspection
```

### 2) Create environment (example)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3) Install core dependencies

```bash
pip install torch torchvision timm optuna pandas scikit-learn pillow pyyaml lightly tqdm numpy openpyxl
```

## Module Usage

### Combination Logic (`CombinationLogicFinal`)

Run from inside `CombinationLogicFinal`:

```bash
cd CombinationLogicFinal
```

#### Hyperparameter Optimization

```bash
python hyperparameter_optimization/hpo.py \
    --study_name <study_name> \
    --storage sqlite:///fixmatch_hpo.db \
    --data_dir <path_to_images> \
    --unlabeled_data_dir <path_to_unlabeled_images> \
    --train_csv <path_to_train_csv> \
    --val_csv <path_to_val_excel> \
    --selfsup_model_path <path_to_ssl_backbone> \
    --output_dir <path_to_output_dir> \
    --n_trials 50
```

#### Train with Best Optuna Trial

```bash
python train/train_combination.py \
    --study_name <study_name> \
    --storage sqlite:///fixmatch_hpo.db \
    --experiment_name <experiment_name> \
    --output_dir <path_to_output_dir> \
    --data_dir <path_to_images> \
    --unlabeled_data_dir <path_to_unlabeled_images> \
    --train_csv <path_to_train_csv> \
    --val_csv <path_to_val_excel> \
    --selfsup_model_path <path_to_ssl_backbone>
```

#### Inference

```bash
python test/inference_combination.py
```

### Self-Supervised Learning (`Self-Supervised-Learning`)

Run from inside `Self-Supervised-Learning`:

```bash
cd Self-Supervised-Learning
```

#### SimCLR Pretraining

```bash
python SSL_Pretrain/simclr.py
```

#### Downstream Supervised Training

```bash
python Training/Train_supervised_downstream.py \
    --expriment_number <run_id> \
    --model efficientnet_v2_s \
    --experiment_name <experiment_name> \
    --train_csv <path_to_train_csv>
```

#### Hyperparameter Optimization

```bash
python Training/Hyperparameter_optimization.py \
    --model_name efficientnet_v2_s \
    --experiment_name <experiment_name> \
    --training_dataset <path_to_training_csv>
```

#### Inference

```bash
python Test/Test.py
```

### Semi-Supervised Learning (`Semi-Supervised-Learning`)

Run from inside `Semi-Supervised-Learning`:

```bash
cd Semi-Supervised-Learning
```

#### Training

```bash
python main.py --config configfiles/configfixmatchdino10.yaml
```

#### Inference

```bash
python testing/inferencedino.py
python testing/inference_efficienet.py
```

## Notes

- Several scripts include hard-coded dataset/model paths; update them for your environment before running.
- Some workflows expect `.xlsx` files (install `openpyxl`, included above).
- For reproducibility, set fixed seeds and keep experiment outputs in dedicated directories.