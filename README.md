### Treinamento no dataset de **árvores**

Exemplo de execução com **1 época**:

```bash
python3 main.py \
  --dataset trees \
  --path-to-dataset trees_large.p \
  --batch-size 1 \
  --log-dir ./results_trees_test \
  --nepochs 1
