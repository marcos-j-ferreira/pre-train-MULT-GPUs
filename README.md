# pre-train-MULT-GPUs

Este repositório contém a implementação de um pré-treinamento utilizando múltiplas GPUs, com **DDP** (Distributed Data Parallel).

Em um projeto recente, publiquei um código para treinar um modelo em uma única GPU. 
[Pré-treinamento - Single GPU](https://github.com/marcos-j-ferreira/pre-train)


Esse treino pode ser feito em várias GPUs, com DDP.

---

## Dataset usado

- [Hugging Face – TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  - Tokenizador BPE
  - ~479 milhões de tokens
  - Vocabulário de 10 mil tokens

- Dataset já tokenizado em `uint16`: [Hugging Face - TinyStories (tokenizado)](https://huggingface.co/datasets/marcos-j-leemes/tinyS/tree/main)

---

## Arquitetura do modelo

Baseado na arquitetura do **GPT-2**.

---

## Ambiente - Kaggle

O **Kaggle** oferece uma versão gratuita com duas GPUs, mas com algumas limitações para rodar o treinamento (tempo de sessão, memória, etc.).

**A principal barreira é a forma como rodar o treinamento**

Para contornar essas limitações, tive que compactar o treino, transformá-lo em um notebook e rodá-lo via comando dentro de uma célula:

```bash
!torchrun --nproc_per_node=2 /kaggle/working/.virtual_documents/__notebook_source__.ipynb
# equivalente a:
# torchrun --nproc_per_node=2 train_ddp.py
```

Dessa forma, consigo rodar o treinamento com múltiplas GPUs a partir de uma única célula. Executar cada célula individualmente **não funciona** para treinamento com múltiplas GPUs — é necessário disparar tudo via `torchrun` em uma célula só.

---

## Requisitos

- Python 3.10+
- PyTorch (com suporte a CUDA)
- 2 ou mais GPUs disponíveis
- `torchrun` (incluso na instalação do PyTorch)

```bash
pip install torch numpy datasets
```

---

## Como rodar

**Localmente (múltiplas GPUs):**

```bash
torchrun --nproc_per_node=<NUM_GPUS> train_ddp.py
```

**No Kaggle:**

```bash
!torchrun --nproc_per_node=2 /kaggle/working/.virtual_documents/__notebook_source__.ipynb
```

---

## Próximos passos / TODO

- [ ] Adicionar checkpoints e retomada de treinamento
- [ ] Suporte a mais de um nó (multi-node, não apenas multi-GPU em um único nó)

---

## Licença
 
MIT 
