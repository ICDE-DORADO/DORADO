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
```

The `data/` directory should contain the benchmark datasets, e.g.,

```text
data/
|-- ICEWS14/
|-- ICEWS18/
|-- ICEWS05-15/
|-- GDELT/
```

## 2. Environment

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

We recommend creating a new virtual environment before installing the dependencies.

## 3. Data Preparation

### 3.1 Build lexical entity-word mappings

For each dataset, first enter the dataset directory and run:

```bash
cd data/ICEWS14
python ent2word.py
cd ../..
```

Replace `ICEWS14` with other dataset names when needed.

### 3.2 Extract BERT entity embeddings

Run the following command to generate BERT-based entity embeddings:

```bash
python get_bert_embeddings.py --dataset ICEWS14 --bert_dir {your_bert_path}
```

Here, `{your_bert_path}` should point to the local directory of the pre-trained BERT model.

### 3.3 Preprocess temporal snapshots

```bash
python data_process.py --dataset ICEWS14 --T 14
```

The argument `--T` denotes the temporal context window used during preprocessing.

## 4. Training and Evaluation

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

## 5. Lifecycle Metric Calculation

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

- **LPI**: Lifelong Performance Index
- **MGI**: Memorization-Generalization Imbalance
- **HSR**: Horizon Sensitivity Rate
