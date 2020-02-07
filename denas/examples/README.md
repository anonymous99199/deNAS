### deNAS experiments

Requirements for running experiments:
* __nas\_benchmarks__ can be found [here](https://github.com/automl/nas_benchmarks/blob/development/README.md)
along with the instructions to download the files.
    * This repo contains both NAS-Bench-101 and NAS-HPO-Bench.
* __nas-bench-1shot1__ can be found [here](https://github.com/automl/nasbench-1shot1).
* __AutoDL-Projects__ can be found [here](https://github.com/D-X-Y/AutoDL-Projects)
    * This repo contains NAS-201.
<br/>
<br/>

Directory structure for the execution of these scripts:
```
..    
|
└───deNAS/   
│   └───denas/
|
└───nas_benchmarks/
│   └───experiment_scripts/
│   └───tabular_benchmarks/
|   |   └───fcnet_benchmark.py
|   |   └───nas_cifar10.py
|   |   └───fcnet_tabular_benchmarks/
|   |   |   └───nasbench_full.tfrecord
|   |   |   └───nasbench_only108.tfrecord
|   |   |   └───fcnet_naval_propulsion_data.hdf5
|   |   |   └───fcnet_protein_structure_data.hdf5
|   |   |   └───fcnet_slice_localization_data.hdf5
|   |   |   └───fcnet_parkinsons_telemonitoring_data.hdf5
|   └───...
|
└───nasbench-1shot1/
│   └───nasbench_analysis/
|   └───optimizers/
|   └───...
|
└───AutoDL-Projects/
│   └───exps/
|   └───libs/
|   └───...
```
