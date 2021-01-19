# [Runmila AI Institute & minoHealth AI Labs Tuberculosis Classification via X-Rays Challenge](https://zindi.africa/competitions/runmila-ai-institute-minohealth-ai-labs-tuberculosis-classification-via-x-rays-challenge)
 ## 11th Place

## Configuration
- For the development of the CNN models, I used an AWS EC2 Instance, more precisely, p3.8xlarge.

| Instance | GPUs              |vCPU | Mem (GiB) |
|----------|-------------------|-----|-----------|
|p3.8xlarge|4x Tesla V100 16GB | 32  | 244       |

## Directory Layout
```
.
├── bin           # Scripts to perform various tasks such as preprocess and train.
├── cache         # Where preprocessed outputs are saved.
├── conf          # Configuration files for classification models.
├── input         # Input files provided by kaggle. 
├── model         # Where classification model outputs are saved.
├── meta          # Where second level model outputs are saved.
├── src           # Code
├── submission    # Submissions (final .csv)
├── notbooks      # Notbooks for inference (This competition was a code competition)
```

## How to
**On the root folder**
### Requirements
- Python 3.6x
- CUDA 11.0
- NVIDIA NGC Pytorch docker image

Build the docker image and pip install the requirements.
```
$ docker build -t <image_name> . && pip install -r requirements.txt
```
### Preprocessing
```
$ sh ./bin/preprocess.sh
```
### Training (classification model)
```
$ sh ./bin/train.sh
```
- Trains all models with 5 folds each

### Predicting (classification model)
```
$ sh ./bin/predict.sh
```
-   Makes predictions for validation data (out-of-fold predictions).
-   Makes predictions for test data.
-   2 bests checkpoints of each fold and each model.

### Second level model (Stacking)
```
$ sh ./bin/predict_meta.sh
```
-   Ensembles out-of-fold predictions from the previous step (used as meta features to construct train data).
-   Ensembles test predictions from the previous step (used as meta features to construct test data).
-   Trains  `LightGBM`,  `Catboost`  and  `XGB`  with 5 folds each.
-   Predicts on test data using each of the trained models.

## Ensemble
```
$ sh ./bin/ensemble.sh
```
- Makes the submission file

