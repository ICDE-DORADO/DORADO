## 1. Project Structure

The project expects the following directory layout:

```text
DORADO/
|-- main.py
|-- model.py
|-- data_process.py
|-- get_bert_embeddings.py
|-- calculate_metrics.py
|-- BERT
|-- data
```

## 2. Environment

- `torch==2.8.0+cu128`
- `torchvision==0.23.0+cu128`
- `torchaudio==2.8.0+cu128`
- `torch-geometric==2.7.0`
- `torch_scatter==2.1.2+pt28cu128`
- `torch_sparse==0.6.18+pt28cu128`
- `torch_cluster==1.6.3+pt28cu128`
- `torch_spline_conv==1.2.2+pt28cu128`

## 3. Data Preparation

```bash
cd data/ICEWS14
python ent2word.py
cd ../..
```

```bash
python get_bert_embeddings.py --dataset ICEWS14 --bert_dir {your_path}
```

```bash
python data_process.py --dataset ICEWS14 --T 14
```

## 4. Train and Test

```bash
python main.py --dataset ICEWS14 --max_length 30 --num_layers 3 --hidden_dim 768 --num_code 50 --patience 3
```

```bash
python main.py --dataset ICEWS18 --max_length 15 --num_layers 2 --hidden_dim 768 --num_code 50 --patience 3
```

```bash
python main.py --dataset ICEWS05-15 --max_length 10 --num_layers 3 --hidden_dim 1024 --num_code 30 --train_history_len 10 --patience 5
```

```bash
python main.py --dataset GDELT --max_length 30 --num_layers 3 --hidden_dim 768 --num_code 100 --train_history_len 7 --patience 3
```

## 5. Calculate Final Metrics

After all split ratios are finished, collect the final result of each split ratio and save them into a text file, for example:

```text
DORADO_ICEWS14.txt
```

File format:

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

This will compute:

- `LPI`
- `MGI`
- `HSR`

