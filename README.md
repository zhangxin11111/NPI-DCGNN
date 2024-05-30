# NPI-DCGNN
NPI-DCGNN transforms NPI detection problems into graph link prediction problems, and uses dual-channel graph neural networks to predict NPI.
    
    Note: The dependent library and version information are in requirements.txt

### 1.1 Generating kmer occurrence vectors

>Python .\src\generate_kmer.py --dataset {datasetName}

#### Parameters

* --dataset : Dataset name. [RPI369 | RPI2241 | RPI3265 | RPI4158 | RPI7317 | NPInter2]

### 1.2 Negative sample selection and partitioning the dataset

>Python .\src\generate_edgelist.py --dataset {datasetName}

#### Parameters

* --dataset : Dataset name. [RPI369 | RPI2241 | RPI3265 | RPI4158 | RPI7317 | NPInter2]

### 1.3 Generating node feature vectors

>Python .\src\generate_node_vec.py --dataset {datasetName} --fold {no}

#### Parameters

* --dataset : Dataset name. [RPI369 | RPI2241 | RPI3265 | RPI4158 | RPI7317 | NPInter2]
* --fold : which fold is this. [0 | 1 | 2 | 3 | 4].  

### 1.4 Generating dataset

>Python .\src\generate_dataset.py --dataset {datasetName} --fold {no}

#### Parameters

* --dataset : Dataset name. [RPI369 | RPI2241 | RPI3265 | RPI4158 | RPI7317 | NPInter2]
* --fold : which fold is this. [0 | 1 | 2 | 3 | 4].  

### 1.5 Training

>Python .\src\train.py --dataset {datasetName} --fold {no}

#### Parameters

* --dataset : Dataset name.[RPI369 | RPI2241 | RPI3265 | RPI4158 | RPI7317 | NPInter2]
* --fold : which fold is this. [0 | 1 | 2 | 3 | 4].  
* --other_parameters : Taking default value. 

## 2. Case study

### 2.1 Generating case study dataset

>Python .\src\case_study\case_study_dataset.py

#### Parameters

* --all_parameters : Taking default value. 

### 2.2 Case study training and prediction

>Python .\src\case_study\case_study_train.py

#### Parameters

* --all_parameters : Taking default value. 




