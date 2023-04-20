# Computation Graphs

Graphs in this directory are example computation graphs that emulates NN training and inference. 

## Checkmate graphs
They are `nx.DiGraph` created from 
[`checkmate.remat.tensorflow2.extraction.dfgraph_from_keras()`](https://github.com/parasj/checkmate/blob/mlsys20_artifact/remat/tensorflow2/extraction.py). 
They were used in the paper *Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization*

## Random layered graphs
They are `nx.DiGraph` created according to Appendix A of *Neural Topological Ordering for Computation Graphs*.
