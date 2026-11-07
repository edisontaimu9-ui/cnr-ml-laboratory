# NLP entity model on an external GPU

1. In the dashboard, open the corpus and download or keep your de-identified JSONL (one document per line: text, entities as [start, end, label], optional language).
2. On a GPU machine (Colab, Kaggle or a rented box): `pip install -r requirements-ner.txt`
3. `python train_ner_gpu.py --data corpus.jsonl --out ner_out --epochs 30` (add `--model <hf-encoder>` to change the encoder, `--cpu` for a small CPU model).
4. Read `ner_out/results.json` (overall and per-language precision, recall, F1) and keep `ner_out/ner_model.zip`.

The split is seeded and nothing is tuned on the test set. Documents with an entity boundary inside a word are skipped and counted.
