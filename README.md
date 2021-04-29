# Fraud-Analysis
**Fraud Analysis** is a Machine Learning Security Python library. This library provides developers and researchers with tools for defending and evaluating Machine Learning models and applications against adversarial attacks.

## Features
- Support For Supervised Learning 
- Inbuild Feature Selection Method
- Testing With Adversarial Attacks
- Implement Adversarial Defence Method

## Requirments
- Python 3

## Usage

### Step 1

Clone this libarary into your project floder

### Step 2

Import the libarary

```python
from main import FAnalysis
```
### Step 3

Initait normal enviornment training

```python
from main import LearningModel

test = FAnalysis("your/dataset/path", LearningModel.RANDOMFOREST, "Lable column name", "attack dataset size(0.1 to 100)")
test.initiate()
test.train_model()
```
#### Learning model type 
- RANDOMFOREST
- LOGISTICREGRESSION 

### Step 4

Generate Attack

```python
from main import Attacks

test.attack(Attacks.ZOO)
```
### Step 5

Implement Defence Method

```python
from main import Defence

test.defence(Defence.ADVERSARIALTRAINING)
```
