# NLP entity model on an external GPU
1. Dashboard > NLP > Corpora > open the corpus > "Train on an external GPU" > Download corpus.
2. On the GPU machine (repo cloned): `pip install -r training/requirements-ner.txt -r api/requirements.txt`
3. `python training/train_ner_gpu.py --repo . --data corpus1-v1.jsonl.gz --out ner_out`
   (`--model <hf-encoder>`, `--cpu`, `--epochs`, `--trial`)
4. Same dashboard card: import `ner_out/results.json` and, to allow registration, `ner_out/ner_model.zip` (max 300 MB).
The run appears in the training runs list marked imported. Use only the de-identified exported corpus.
