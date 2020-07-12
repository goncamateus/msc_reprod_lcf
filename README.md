# Previsão de potência eólica de curtíssimo prazo baseada na análise espectral e decomposição da série temporal
Reproduction of [Previsão de potência eólica de curtíssimo prazo baseada na análise espectral e decomposição da série temporal](https://repositorio.ufpe.br/bitstream/123456789/32495/1/DISSERTA%c3%87%c3%83O%20Lucas%20Cabral%20Fernandes.pdf) in Python.


## Requirements
```shell
$pip install -r requirements.txt 
```

## Usage

- dataset: 
    
    - The input time serie you want to fit in our model

- decompose:

    - Rather you wanna decompose or not your serie
    - Set it 1 if you did not decomposed your serie yet

- test:

    - Rather you wanna train or not your model
    - Set it 0 if you did not created and trained a Model

- regvars:
    
    - How many regressor variables you want for your model(s)
    - Default is 60

- horizons:
    
    - How many horizons you want to predict
    - Default is 12

- models:

    - Which DNN models you wanna try.
    - Choose any combination of MLP, GRU, LSTM

- epochs:

    - Number of epochs you want to train each model.
    - Default is 3

```shell
$python main.py --dataset data/inputs/your_serie --decompose 1 --test 0 --regvars 60 --horizons 12 --models mlp,gru,lstm --epochs 3
```
