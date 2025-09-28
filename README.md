# DGMG – Experimentos com Ordenações de Nós (Cycles)

Este diretório contém o pipeline para:
1) **gerar ciclos** como matrizes de adjacência,
2) **converter** cada grafo para **sequências de decisão** do DGMG usando **diferentes ordenações de nós**,
3) **treinar/avaliar** um modelo por ordenação,
4) **comparar** as métricas de avaliação (qual ordenação gera ciclos com maior assertividade).

> Base: _Learning Deep Generative Models of Graphs_ (Li et al., 2018), com datasets sintéticos de ciclos.

---

## Estrutura

```
dgmg/
  build_cycles_datasets.py   # gera .p por ordenação (lista de sequências DGMG)
  train_by_order.py          # treina/avalia um modelo por ordenação
  compare_eval.py            # lê model_eval.txt e compara as ordenações
  main.py                    # treino/avaliação do DGMG
  model.py, utils.py, cycles.py, trees.py
  sample_large.py            # amostra 1 grafo (rollout inference)
  sample_many.py             # amostra N grafos e sumariza tamanhos
  runs/                      # saídas (datasets, checkpoints, avaliação)
gds/
  get_decision.py            # ordenações + conversor Adjacência -> Sequência
```

---

## Requisitos

Crie e ative um ambiente, depois instale as dependências:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## 1) Gerar datasets por ordenação

Gera ciclos aleatórios (tamanho entre `min-size` e `max-size`) e cria um **.p por ordenação** com a sequência de decisões compatível com o `CycleDataset`.

```bash
cd dgmg

python3 build_cycles_datasets.py   --min-size 5 --max-size 20   --num-graphs 4000   --orderings degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral   --outdir ./runs/datasets
```

Saída esperada (exemplo):
```
./runs/datasets/
  cycles_degree.p
  cycles_bfs.p
  cycles_dfs.p
  ...
```

> As ordenações disponíveis vêm de `gds/get_decision.py`:  
> `degree`, `bfs`, `dfs`, `degeneracy`, `random`, `mcs`, `lexbfs`, `spectral`.

---

## 2) Treinar e avaliar por ordenação

Para cada `.p` gerado, chamamos `main.py` com `--dataset cycles` apontando para o arquivo correspondente:

```bash
python3 train_by_order.py   --datasets-dir ./runs/datasets   --orderings degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral   --batch-size 1   --nepochs 25   --logroot ./runs
```

Isto cria uma árvore como:

```
runs/
  train_orders_2025-09-15_21-03-11/
    degree/2025-09-15_21-03-12/model_eval.txt
    bfs/   2025-09-15_21-03-35/model_eval.txt
    ...
```

Cada `model_eval.txt` contém (exemplo):

```
num_samples    10000
v_min          5
v_max          20
average_size   12.34
valid_size_ratio 0.9980
cycle_ratio    0.9850
valid_ratio    0.9830
```

- **cycle_ratio**: fração de amostras geradas que são ciclos válidos.  
- **valid_size_ratio**: fração de grafos dentro de `[v_min, v_max]`.  
- **valid_ratio**: fração que é **(ciclo) E (tamanho válido)**.  
- **average_size**: tamanho médio gerado.

---

## 3) Comparar ordenações

Após os treinos acima, rode:

```bash
python3 compare_eval.py
```

Saída (exemplo):

```
Usando: ./runs/train_orders_2025-09-15_21-03-11

ordering      average_size   valid_size_ratio   cycle_ratio   valid_ratio
bfs               12.4100             0.9990        0.9920        0.9910
degree            12.3800             0.9993        0.9875        0.9869
dfs               12.4200             0.9991        0.9904        0.9899
degeneracy        12.4000             0.9992        0.9931        0.9925
...
```

Escolha a ordenação com melhor **`cycle_ratio`** (e `valid_ratio`) para o seu caso.

---

## 4) Amostragem

### 4.1 Uma amostra (inference “puro”)
Carregue um checkpoint e gere **1** grafo. `--num-nodes` define o **teto** (`v_max`) permitido na geração – o modelo **pode** parar antes.

```bash
python3 sample_large.py   --ckpt ./runs/<timestamp>/model.pth   --num-nodes 100   --save-plot ./runs/sample_cycle_N100.png
```

> Observação: mudar o teto **não obriga** a gerar 100 nós; o agente aprendeu a **parar** por volta da faixa de treino.

### 4.2 Muitas amostras (estatística de tamanhos)
```bash
python3 sample_many.py   --ckpt ./runs/<timestamp>/model.pth   --trials 400   --v-max 100   --threshold 20   --save-dir ./runs/over20_pngs   --save-limit 30
```

Mostra média, máximo observado e quantas amostras passaram de 20 nós.

### 4.3 (Opcional) Gerar **ciclo exato** com `N` nós (teacher-forcing)
Para ciclos você pode **garantir N exato** executando a sequência canônica `get_decision_sequence(N)` (teacher-forcing).  
Se quiser, crie um `sample_teacher.py` que use `cycles.get_decision_sequence(N)` e aplique `model(actions=...)` com `torch.no_grad()` — isso sempre produz um ciclo perfeito com `N`.

---

## 5) FAQ / Dicas

- **Por que várias ordenações?**  
  O DGMG é sequencial: a **ordem** define o contexto das decisões. Algumas ordens (ex.: `bfs`, `degeneracy`) tendem a estabilizar as decisões de arestas e melhorar `cycle_ratio`.

- **Dataset compatível**  
  Os `.p` gerados aqui são **listas de sequências** no mesmo formato do `CycleDataset`, então `main.py --dataset cycles --path-to-dataset <arquivo>.p` funciona direto.

- **Spectral sem SciPy**  
  A ordenação `spectral` foi implementada via NumPy denso (ok para N≤20), evitando dependência do SciPy.

- **Reprodutibilidade**  
  Scripts usam `--seed` para desempates estocásticos e o `utils.py` já fixa seeds do PyTorch/Random.

---

## 6) Resultados típicos (guia de leitura)

- Experimentos com ciclos 5–20 normalmente resultam em `cycle_ratio` e `valid_ratio` **próximos de 1.0** nas melhores ordenações.
- **Mudar `v_max`** (ex.: 100) **só permite** crescer; não força. Se quer **N exato**, use teacher-forcing com a sequência canônica.

---

## 7) Comandos rápidos (TL;DR)

```bash
# 1) Gerar datasets por ordenação
python3 build_cycles_datasets.py --min-size 5 --max-size 20 --num-graphs 4000 --outdir ./runs/datasets

# 2) Treinar por ordenação
python3 train_by_order.py --datasets-dir ./runs/datasets --nepochs 25 --logroot ./runs

# 3) Comparar avaliações
python3 compare_eval.py

# 4) Amostrar 1 grafo (teto 100)
python3 sample_large.py --ckpt ./runs/<timestamp>/model.pth --num-nodes 100 --save-plot ./runs/sample_N100.png
```

---

## 8) Créditos

Implementação inspirada por **Li et al., 2018** – *Learning Deep Generative Models of Graphs*.  
Componentes auxiliares (ordenadores + conversor), em `gds/get_decision.py`.
