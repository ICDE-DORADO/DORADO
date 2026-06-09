import os
import argparse
import numpy as np
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModel


def load_index(path):
    names = {}
    last_err = None
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 2:
                        continue
                    name, idx = parts
                    try:
                        idx = int(idx)
                    except ValueError:
                        continue
                    names[idx] = name
            break
        except UnicodeDecodeError as err:
            names.clear()
            last_err = err
            continue
    else:
        if last_err is not None:
            raise last_err
    return [names[i] for i in sorted(names.keys())]


@torch.no_grad()
def encode_texts(text_list, tokenizer, model, device, batch_size=64, max_length=64):
    model.eval()
    all_vecs = []

    for i in tqdm(range(0, len(text_list), batch_size), desc="BERT encoding"):
        batch = text_list[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        all_vecs.append(cls_vec.cpu())

    all_vecs = torch.cat(all_vecs, dim=0)
    return all_vecs.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ICEWS14",
                        help="Dataset name, e.g., ICEWS14, ICEWS05-15")
    parser.add_argument("--bert_dir", type=str, default="BERT/bert-base-uncased",
                        help="Local BERT model directory (relative to project root)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory (default: ../adhub/ICEWS14/0.1/data/ICEWS14 for server)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "code" else script_dir

    if args.data_dir is None:
        candidate_dirs = [
            os.path.join(project_root, "adhub", "ICEWS14", "0.1", "data", args.dataset),
            os.path.join(project_root, "adhub", args.dataset, "0.1", "data", args.dataset),
            os.path.join(script_dir, "data", args.dataset),
            os.path.join(project_root, "data", args.dataset),
        ]
        data_dir = next((path for path in candidate_dirs if os.path.isdir(path)), candidate_dirs[-1])
    else:
        data_dir = args.data_dir
    entity_path = os.path.join(data_dir, "entity2id.txt")
    relation_path = os.path.join(data_dir, "relation2id.txt")
    word_path = os.path.join(data_dir, "word2id.txt")

    assert os.path.exists(entity_path), f"entity2id not found: {entity_path}"
    assert os.path.exists(relation_path), f"relation2id not found: {relation_path}"
    assert os.path.exists(word_path), f"word2id not found: {word_path}"

    print(f"[Info] Loading indices from {data_dir}")
    entities = load_index(entity_path)
    relations = load_index(relation_path)
    words = load_index(word_path)

    print(f"[Info] #entities={len(entities)}, #relations={len(relations)}, #words={len(words)}")

    bert_path = os.path.join(script_dir, args.bert_dir)
    if not os.path.isdir(bert_path):
        bert_path = os.path.join(project_root, args.bert_dir)
    print(f"[Info] Loading BERT from {bert_path}")
    assert os.path.isdir(bert_path), f"BERT dir not found: {bert_path}"
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    model = AutoModel.from_pretrained(bert_path, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

               
    print("\n[Step 1] Encoding entities...")
    ent_emb = encode_texts(entities, tokenizer, model, device,
                           batch_size=args.batch_size, max_length=args.max_length)
    ent_out_path = os.path.join(data_dir, f"{args.dataset}_Bert_Entity_Embedding.npy")
    np.save(ent_out_path, ent_emb)
    print(f"[Done] Entity embeddings saved to {ent_out_path}, shape={ent_emb.shape}")

               
    print("\n[Step 2] Encoding words...")
    word_emb = encode_texts(words, tokenizer, model, device,
                            batch_size=args.batch_size, max_length=args.max_length)
    word_out_path = os.path.join(data_dir, f"{args.dataset}_Bert_Word_Embedding.npy")
    np.save(word_out_path, word_emb)
    print(f"[Done] Word embeddings saved to {word_out_path}, shape={word_emb.shape}")

               
    print("\n[Step 3] Encoding relations...")
    rel_emb = encode_texts(relations, tokenizer, model, device,
                           batch_size=args.batch_size, max_length=args.max_length)
    rel_out_path = os.path.join(data_dir, f"{args.dataset}_Bert_Relation_Embedding.npy")
    np.save(rel_out_path, rel_emb)
    print(f"[Done] Relation embeddings saved to {rel_out_path}, shape={rel_emb.shape}")

    print("\nAll done.")


if __name__ == "__main__":
    main()


