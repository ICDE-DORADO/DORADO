# DORADO

This repository contains the implementation of **DORADO**:  
**Dual-Stream Complementary Mutual Distillation for Generalized Full-Lifecycle Temporal Knowledge Graph Reasoning**.

DORADO is designed for temporal knowledge graph reasoning under continuously changing historical observability. It integrates a lifecycle-memory transductive stream and a lifecycle-generalization inductive stream, coordinated by dynamic mutual distillation.

## 1. Project Structure

The expected directory structure is:

```text
DORADO/
|-- main.py                  # Training and evaluation entry
|-- model.py                 # DORADO model implementation
|-- data_process.py          # Temporal data preprocessing
|-- get_bert_embeddings.py   # BERT-based entity embedding extraction
|-- calculate_metrics.py     # Lifecycle metric calculation
|-- BERT/                    # Pre-trained BERT model directory
|-- data/                    # Dataset directory
````

The `data/` directory should contain the benchmark datasets, e.g.,

```text
data/
|-- ICEWS14/
|-- ICEWS18/
|-- ICEWS05-15/
|-- GDELT/
```

## 2. Benchmark Datasets and Evaluation Protocol

DORADO is evaluated on four temporal knowledge graph benchmarks: **ICEWS14**, **ICEWS18**, **ICEWS05-15**, and **GDELT**.

To evaluate model robustness under continuously changing historical observability, we use a **Continuous Evolutionary Evaluation Protocol**. Instead of relying on a single fixed train/validation/test split, this protocol constructs multiple chronological split ratios:

```text
5:45:50, 10:40:50, 20:40:40, 30:30:40, 40:30:30, 50:20:30, 60:20:20, 70:10:20, 80:10:10
```

For each split, entities are categorized according to whether they have appeared in the observable historical facts before the query time:

* **Stationary entities (`# Stat.`)**: entities that have appeared in the observable history.
* **Emerging entities (`# Emerg.`)**: entities with no prior observations in the observable history.
* **Temporal observation ratio (`alpha`)**: the ratio between stationary and emerging entities, computed as `# Stat. / # Emerg.`.

A smaller `alpha` indicates a cold-start stage dominated by emerging entities, while a larger `alpha` indicates a mature stage dominated by stationary entities.

The detailed split statistics are listed below.

### ICEWS14

| Train:Val:Test | # Stat. | # Emerg. |  alpha |
| -------------- | ------: | -------: | -----: |
| 5:45:50        |   1,426 |    5,702 |  0.250 |
| 10:40:50       |   2,248 |    4,880 |  0.461 |
| 20:40:40       |   3,398 |    3,730 |  0.906 |
| 30:30:40       |   4,132 |    2,996 |  1.379 |
| 40:30:30       |   4,750 |    2,378 |  1.997 |
| 50:20:30       |   5,274 |    1,854 |  2.845 |
| 60:20:20       |   5,713 |    1,415 |  4.037 |
| 70:10:20       |   6,081 |    1,047 |  5.808 |
| 80:10:10       |   6,480 |      640 | 10.125 |

### ICEWS18

| Train:Val:Test | # Stat. | # Emerg. |  alpha |
| -------------- | ------: | -------: | -----: |
| 5:45:50        |   5,119 |   17,914 |  0.296 |
| 10:40:50       |   8,032 |   15,001 |  0.535 |
| 20:40:40       |  11,776 |   11,257 |  1.046 |
| 30:30:40       |  14,310 |    8,723 |  1.640 |
| 40:30:30       |  16,292 |    6,741 |  2.417 |
| 50:20:30       |  17,909 |    5,124 |  3.495 |
| 60:20:20       |  19,133 |    3,900 |  4.906 |
| 70:10:20       |  20,263 |    2,770 |  7.315 |
| 80:10:10       |  21,186 |    1,847 | 11.470 |

### ICEWS05-15

| Train:Val:Test | # Stat. | # Emerg. | alpha |
| -------------- | ------: | -------: | ----: |
| 5:45:50        |   2,988 |    7,500 | 0.398 |
| 10:40:50       |   4,095 |    6,393 | 0.641 |
| 20:40:40       |   5,578 |    4,910 | 1.136 |
| 30:30:40       |   6,558 |    3,930 | 1.669 |
| 40:30:30       |   7,343 |    3,145 | 2.335 |
| 50:20:30       |   7,931 |    2,557 | 3.102 |
| 60:20:20       |   8,433 |    2,045 | 4.124 |
| 70:10:20       |   8,895 |    1,593 | 5.584 |
| 80:10:10       |   9,306 |    1,182 | 7.873 |

### GDELT

| Train:Val:Test | # Stat. | # Emerg. |  alpha |
| -------------- | ------: | -------: | -----: |
| 5:45:50        |   2,845 |    4,846 |  0.587 |
| 10:40:50       |   3,840 |    3,851 |  0.997 |
| 20:40:40       |   4,765 |    2,926 |  1.629 |
| 30:30:40       |   5,349 |    2,342 |  2.284 |
| 40:30:30       |   5,937 |    1,754 |  3.385 |
| 50:20:30       |   6,234 |    1,457 |  4.279 |
| 60:20:20       |   6,655 |    1,036 |  6.424 |
| 70:10:20       |   6,905 |      786 |  8.785 |
| 80:10:10       |   7,230 |      461 | 15.683 |

## 3. Environment

The code is tested with the following environment:

```text
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
torch-geometric==2.7.0
torch_scatter==2.1.2+pt28cu128
torch_sparse==0.6.18+pt28cu128
torch_cluster==1.6.3+pt28cu128
torch_spline_conv==1.2.2+pt28cu128
```

We recommend creating a new virtual environment before installing the dependencies. Please make sure that the PyTorch Geometric packages are compatible with your PyTorch and CUDA versions.

## 4. Data Preparation

### 4.1 Build lexical entity-word mappings

For each dataset, first enter the dataset directory and run:

```bash
cd data/ICEWS14
python ent2word.py
cd ../..
```

Replace `ICEWS14` with other dataset names when needed.

### 4.2 Extract BERT entity embeddings

Run the following command to generate BERT-based entity embeddings:

```bash
python get_bert_embeddings.py --dataset ICEWS14 --bert_dir {your_bert_path}
```

Here, `{your_bert_path}` should point to the local directory of the pre-trained BERT model.

### 4.3 Preprocess temporal snapshots

```bash
python data_process.py --dataset ICEWS14 --T 14
```

The argument `--T` denotes the temporal context window used during preprocessing.

## 5. Training and Evaluation

Use `main.py` to train and evaluate DORADO. Example commands for each dataset are listed below.

### ICEWS14

```bash
python main.py --dataset ICEWS14 --max_length 30 --num_layers 3 --hidden_dim 768 --num_code 50 --patience 3
```

### ICEWS18

```bash
python main.py --dataset ICEWS18 --max_length 15 --num_layers 2 --hidden_dim 768 --num_code 50 --patience 3
```

### ICEWS05-15

```bash
python main.py --dataset ICEWS05-15 --max_length 10 --num_layers 3 --hidden_dim 1024 --num_code 30 --train_history_len 10 --patience 5
```

### GDELT

```bash
python main.py --dataset GDELT --max_length 30 --num_layers 3 --hidden_dim 768 --num_code 100 --train_history_len 7 --patience 3
```

## 6. Lifecycle Metric Calculation

After finishing all split-ratio experiments, collect the final results of each split ratio into a text file, for example:

```text
DORADO_ICEWS14.txt
```

The file should follow the format below:

```text
[5,45,50]
MRR: 0.1234, Hit1: 0.0567, Hit3: 0.1456, Hit10: 0.2345, Emerging_MRR: 0.1111, Emerging_Hit1: 0.0500, Emerging_Hit3: 0.1300, Emerging_Hit10: 0.2200, Known_MRR: 0.1350, Known_Hit1: 0.0600, Known_Hit3: 0.1500, Known_Hit10: 0.2400
[10,40,50]
MRR: 0.1250, Hit1: 0.0570, Hit3: 0.1470, Hit10: 0.2360, Emerging_MRR: 0.1130, Emerging_Hit1: 0.0510, Emerging_Hit3: 0.1320, Emerging_Hit10: 0.2230, Known_MRR: 0.1370, Known_Hit1: 0.0610, Known_Hit3: 0.1520, Known_Hit10: 0.2420
```

Then run:

```bash
python calculate_metrics.py DORADO_ICEWS14.txt
```

This script computes the lifecycle robustness metrics:

* **LPI**: Lifelong Performance Index
* **MGI**: Memorization-Generalization Imbalance
* **HSR**: Horizon Sensitivity Rate

In the result file, `Emerging_*` denotes the performance on emerging entities, while `Known_*` denotes the performance on stationary entities.


