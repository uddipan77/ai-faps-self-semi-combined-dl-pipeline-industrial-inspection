## A Comparative Assessment of Self- and Semi Supervised Learning as Well as Combined Approaches for Deep Learning Based Image Classification in Industrial Visual Inspection

```
ai-faps-shouvik-chattopadhyay/
├── CombinationLogicFinal/
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── hyperparameter_optimization/
│   │   └── hpo.py
│   ├── test/
│   │   ├── __init__.py
│   │   └── inference_combination.py
│   ├── train/
│   │   ├── __init__.py
│   │   └── train_combination.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   └── manualusedutils.py
├── Self-Supervised Learning/
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   └── Dataset.py
│   │   ├── modeling/
│   │   │   ├── __init__.py
│   │   │   ├── make_model.py
│   │   │   └── train_validation_test.py
│   │   ├── SSL_Pretrain/
│   │   │   └── simdclr.py
│   │   ├── Test/
│   │   │   └── Test.py
│   │   ├── Training/
│   │   │   ├── Hyperparameter_optimization.py
│   │   │   └── Train_supervised_downstream.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── Utils.py
├──Semi-Supervised-Learning/
├── configfiles/
│   ├── configfixmatchdino10.yaml
│   ├── configfixmatchdino25.yaml
│   ├── configfixmatchdino50.yaml
│   ├── configfixmatchdino100.yaml
│   ├── configfixmatchefficient10.yaml
│   ├── configfixmatchefficient25.yaml
│   ├── configfixmatchefficient50.yaml
│   ├── configfixmatchefficient100.yaml
├── dataset/
│   ├── __init__.py
│   └── datasets.py
├── models/
│   ├── __init__.py
│   ├── customdnnmodel.py
│   ├── customefficientnet.py
│   ├── custommodel.py
│   ├── custommodelefficient.py
│   └── inferenceefficient.py
├── train/
│   ├── __init__.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── manualusedutils.py
│   └── mimatchutils.py
├── main.py
├── .gitignore
````

## Usage
```
git clone https://github.com/andi677/ai-faps-shouvik-chattopadhyay.git
cd ai-faps-shouvik-chattopadhyay

````

## Installation Instructions

### Create and Activate the Conda Environment

1. **Ensure Conda is installed**: If Conda is not installed, download and install it from the [official Anaconda website](https://www.anaconda.com/products/individual).

2. **Create the Environment**: Run the following command to create the environment from the `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate env1
```


### Combination Logic

#### Training

To start training, load the hyperparameters from the sqlite file. Then use the following commands:

```python
python3 train_combination.py \
    --experiment_name <experiment_name> \
    --output_dir <path_to_output_directory> \
    --study_name <study_name> \
    --storage <database_connection_string> \
    --data_dir <path_to_supervised_data> \
    --unlabeled_data_dir <path_to_unlabeled_data> \
    --train_csv <path_to_train_csv> \
    --val_csv <path_to_validation_csv> \
    --selfsup_model_path <path_to_pretrained_selfsupervised_model>

````
#### Hyperparameter Optimization

The search space for hyperparameter tuning has already been defined. To run the optimization, simply load the model and adjust the training hyperparameters if required. Then, execute the script:

```python

python3 hpo.py \
    --data_dir <path_to_supervised_data> \
    --unlabeled_data_dir <path_to_unlabeled_data> \
    --train_csv <path_to_train_csv> \
    --val_csv <path_to_validation_csv> \
    --selfsup_model_path <path_to_pretrained_selfsupervised_model> \
    --output_dir <path_to_output_directory> \
    --n_trials <number_of_trials> \
    --study_name <study_name> \
    --storage <database_connection_string> \
    --direction <maximize_or_minimize>

````

#### Inference ####

To run the inference, execute the script:

```python

python3 inference_combination.py

````

### Self-Supervised-Learning

#### Pretraining

To start pretraining, use the following commands:

```python

python3 SSL_Pretrain/simclr.py

````

#### Downstream Process

To start downstream training, use the following commands:

1. **Load Hyperparameters:**
- In the `Train_supervised_downstream.py` script, load the best hyperparameters recommended by the Optuna study into the `config_dict`. Configure any additional training hyperparameters as needed.
2. **Run Training:**
- Execute the script:
```python
python3 Training/Train_supervised_downstream.py

````

#### Hyperparameter Optimization

The search space for hyperparameter tuning has already been defined.  To run the optimization, simply load the backbone model and adjust the training hyperparameters within the script. Then, execute the script:
```python
# For HYO of SL trained and SSL pretrained models
python3 Training/Hyperparameter_optimization.py

````

#### Inference ####

To run the inference, execute the script:

```python

python3 Test/Test.py

````

### Semi-Supervised-Learning

#### Training ####

To start training, use the main.py and the path to the specific config file to run. All hyperparameters from the optuna as well as the path for the data, labels and other parameters are to be changed in the config file.

```python

python3 multilabel/main.py --config /pathtoconfig.yaml

````

#### Inference ####

To run inference of the model on the test data, copy the path of the best model and paste in the best_model_path for the respective backbone model. Then run

```python
python3 multilabel/testing/inferencedino.py
````