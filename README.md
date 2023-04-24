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

## Reproducing the paper

Run the experiments
```
python3 main.py -g "data/random_layered_n100_w0.27_nlv0.75_ed0.2_scd0.14.json" -m 0.9 -o "output/icml"
python3 main.py -g "data/random_layered_n100_w0.27_nlv0.75_ed0.2_scd0.14.json" -m 0.8 -o "output/icml"
python3 main.py -g "data/random_layered_n250_w0.43_nlv0.75_ed0.2_scd0.14.json" -m 0.9 -o "output/icml"
python3 main.py -g "data/random_layered_n250_w0.43_nlv0.75_ed0.2_scd0.14.json" -m 0.8 -o "output/icml"
python3 main.py -g "data/random_layered_n500_w0.36_nlv0.75_ed0.2_scd0.14.json" -m 0.9 -o "output/icml"
python3 main.py -g "data/random_layered_n500_w0.36_nlv0.75_ed0.2_scd0.14.json" -m 0.8 -o "output/icml"
python3 main.py -g "data/random_layered_n1000_w0.31_nlv0.75_ed0.2_scd0.14.json" -m 0.9 -o "output/icml"
python3 main.py -g "data/random_layered_n1000_w0.31_nlv0.75_ed0.2_scd0.14.json" -m 0.8 -o "output/icml"
python3 main.py -g "data/ResNet50 (MLSys)_256_(224, 224, 3)_train_nx.json" -m 0.9 -o "output/icml"
python3 main.py -g "data/ResNet50 (MLSys)_256_(224, 224, 3)_train_nx.json" -m 0.8 -o "output/icml"
python3 main.py -g "data/fcn_8_vgg (MLSys)_32_(416, 608, 3)_train_nx.json" -m 0.9 -o "output/icml"
python3 main.py -g "data/fcn_8_vgg (MLSys)_32_(416, 608, 3)_train_nx.json" -m 0.8 -o "output/icml"
```
Parse the results
```
python3 print_latex_table.py -o "output/icml"
```
