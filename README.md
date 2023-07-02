# TGIF
Graph-based fasion recommendation

# How to run 

## dataset
- Put all original dataset into `dataset` directory

## Docker Container
- Docker container use cgmc project directory as volume 
- File change will be apply directly to file in docker container

## Preprocessing
1. `make up` : build docker image and start docker container
3. `python3 src/preprocessing.py` : start data preprocessing in docker container

## Train 
1. `make up` : build docker image and start docker container
2. check `train_config/train_list.ymal` file
3. `python3 src/train.py` : start train in docker container

## Test
1. `make up` : build docker image and start docker container
2. check `test_config/test_list.ymal` file
3. `python3 src/test.py` : start train in docker container

<br />