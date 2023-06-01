# Sample Code
This code is an implementation of REINFORCEMENT algorithm for EE619 project 3.

## Getting Started
### step 1. Create your virtual env. 
For this, we recommend [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

### step 2. Install requirements
```
pip install -r requirements.txt
```
If the command doesn't work,
```
pip install -r requirements.txt --no-cache-dir
```
If you want to use different versions from the requirements.txt, first install "dm_control" & "torch",
```
pip install wheel dm_control torch
```
then you should install other requirements.

### step 3. Run & Evaluate the scripts
For training,
```
python train.py --save_path=trained_model.pt
```

For evaluating,
```
python evaluate.py
```
