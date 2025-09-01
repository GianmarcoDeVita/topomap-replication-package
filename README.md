# Replication package for the research paper "TopoMap: A Feature-based Semantic Discriminator of the Topographical Regions in the Test Input Space"

This repository contains all the code needed to replicate the results in the aforementioned research paper. 
There is a folder for each application subject.

## Pipeline1 - Cluster Generation and Evaluation Pipeline 

For each subject XXXX, within folder ``Pipeline1``, the cluster generation and evaluation pipeline is implemented in a file named ``XXXX_generate_map.py``. The trained DNN models for evaluating have been saved at each empirical iteration, but due to the repository restrictions, they may be available upon request.

## Pipeline2 - Evaluation Based on Mutation Analysis

For each subject XXXX, within folder ``Pipeline2``:
  - file ``XXXX_mutants.json`` contains a dictionary listing the mutants that are killable, hence considered in the experiments.  
  - file ``misclassification_distribution/XXXX_misclassification_distribution.py`` computes the raw number of contributors in the entire test set for each killable mutant, and assesses whether such mutant is killable in a statistically significant way by the test set taken alone.
  - the pipeline conducting the mutation analysis is implemented in ``XXXX_cluster_mutation_analysis.py``.
  - the pipeline implemented in ``XXXX_cluster_mutation_analysis_random.py`` performs a mutation analysis on a random selection of inputs the same size of the killing aggregation found before.
  - the script included in file ``killing_propagation.py`` computes metrics useful to answer RQ3.
  - the script included in ``get_topology/get_topology.py`` produces a two-panel plot, where the left panel is a graphical spatial depiction of the test set, obtained through t-SNE and highlighting the different clusters, while the right panel includes a heatmap showing, for each mutant, the clusters included in the aggregation killing that mutant.

The trained original and mutated models are taken from the DeepCrime replication package, which is publicly available online on Zenodo. For that, we refer to the reference included in the research paper.

## RQ4 - Killing Propagation

Folder ``RQ4`` contains the material and reports for all subjects used to answer RQ4.
