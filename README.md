# Askonas 

Askonas is a threat detection architecture that is inspired by the biological immune system. It uses machine learning to leverage detection through Fed-Active learning.

Stages completed:
- Architecture design
- Fed-Active model
- Learning methods performance analysis

Next stages:
- Code refactoring and orgaization
- Test coverage

## Dataset
The datasets used were CIC-IDS-2018 and CIC-IDS-2017

## How to run this project

At the current version, this repository contains only the code for data preparation and model training. 
To run the dataset preparation routines:
```
python main_dataset_generate.py
```

To run the basic training routines:
```
python main_trainer.py
```

To run Federated Learning training routines:
```
./federated/run.sh
```

To run the active learning inspired dataset filtering:
```
python active_learning/uncertainty.py
```

For Fed-Active, filter dataset using the Active leraning inpsired filter and retrain through federated learning.
