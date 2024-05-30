# TGIF
## Less is More: A Streamlined Graph-Based Fashion Outfit Recommendation without Multimodal Dependency
## SAC 2024
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


## Citation
```
@inproceedings{less_is_more_2024
author = {Kim, Daehee and Han, Donghee and Roh, Daeyoung and Han, Keejun and Yi, Mun Yong},
title = {Less is More: A Streamlined Graph-Based Fashion Outfit Recommendation without Multimodal Dependency},
year = {2024},
isbn = {9798400702433},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3605098.3636168},
doi = {10.1145/3605098.3636168},
series = {SAC '24}
}
```

<br />
