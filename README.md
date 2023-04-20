# Mocassin

This repo implements Mocassin, a constraint programming (CP) method for rematerialization. 
It makes use of OR-Tools' [CP-SAT Solver](https://developers.google.com/optimization/cp/cp_solver).

## Installation

Download the repo
```
git clone https://github.com/haoming-codes/mocassin.git
cd mocassin
```
Create the environment (e.g. using conda)
```
conda create -n mocassin_env python=3.9
conda activate mocassin_env
pip3 install .
```


## Running experiments

Run the experiments
```
python3 main.py
```
Parsing the results
```
python3 parse.py
```
