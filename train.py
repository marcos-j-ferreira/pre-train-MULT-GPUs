"""
Treinamento com múltiplas GPUs usando DistributedDataParallel (DDP).

Rodar com:
    torchrun --nproc_per_node=2 train_ddp.py

Principais mudanças em relação à versão de sigle GPU:
  1. Multiprocessing real: um processo por GPU (torchrun cria e gerencia isso).
  2. init_process_group + destroy_process_group no início/fim.
  3. rank / local_rank / world_size lidos das variáveis de ambiente que o torchrun injeta.
  4. Cada processo usa sua própria GPU (device = cuda:{local_rank}), não mais um único
     device controlando todas.
  5. get_batch agora restringe o sorteio de índices à fatia lógica do dataset que
     pertence a este rank (equivalente manual a um DistributedSampler).
  6. Modelo empacotado com DDP(model, device_ids=[local_rank]) em vez de
     nn.DataParallel(model). O DDP sincroniza gradientes via all-reduce
     automaticamente durante o loss.backward().
  7. BATCH_SIZE agora é o batch LOCAL de cada GPU (não é mais fatiado como no
     DataParallel). O batch efetivo global passa a ser BATCH_SIZE * world_size.
  8. Prints, evaluate() e save_model() só rodam no rank 0, para não duplicar
     logs/checkpoints (cada processo faria isso senão).
  9. Removido o DebugModel/prints de shape por GPU — não é mais necessário, já
     que agora sabemos exatamente o que cada rank recebe (seu batch completo).
"""

# imports
from main import ModeloCompleto
from main import Config
from main import get_lr
from main import save_model

# imports library
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import numpy as np

# Dados
DATASET = os.path.dirname(os.path.abspath(__file__)) # Essa configuração da erro no Kaggle alterar para:
# DATASET = "" # Essa forma não dar erro
FILENAME = "train.bin" 

# Configurações do modelo
VOCAB_SIZE = 10001
EMBEDDING_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2
BLOCK_SIZE = 100
DROPOUT = 0.0

# configurações do treinamento
MAX_STEPS = 100
BATCH_SIZE = 12          # <-- agora é o batch LOCAL por GPU (não fatiado)
GRAD_ACCUM_STEPS = 1
WEIGHT_DECAY = 0.0001
WARMUP_STEPS = 10
LEARNING_RATE = 1e-3
MIN_LR = 1e-5
GRAD_CLIP = 1.0

# logs
PRINT_INTERVAL = 1
EVAL_INTERVAL = 100

# Configuração do hardware
USE_AMP = False
USE_BF16 = False
TORCH_COMPILE = False

# save model
STEP_SAVE_INTERVAL = 500
SAVE_DIR = "model_final.pth"


# ---------------------------------------------------------------------------
# FEAT 1/2/3: setup do process group + leitura de rank/local_rank/world_size
# ---------------------------------------------------------------------------
def ddp_setup():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def ddp_cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FEAT 5: get_batch agora recebe rank/world_size e sorteia só dentro da
# fatia lógica do dataset que pertence a este processo.
# ---------------------------------------------------------------------------
data_dir = DATASET
def get_batch(split, batch_size, block_size, device, rank, world_size):
    filename = FILENAME if split == 'train' else 'val.bin'
    path = os.path.join(data_dir, filename)
    if split == 'val' and not os.path.exists(path):
        path = os.path.join(data_dir, FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")

    data = np.memmap(path, dtype=np.uint16, mode='r')
    total_len = len(data) - block_size
    if total_len <= 0:
        raise ValueError(f"{path} possui poucos tokens para BLOCK_SIZE={block_size}.")

    # --- divisão lógica do dataset entre os ranks (sem shards físicos) ---
    shard_len = total_len // world_size
    shard_start = rank * shard_len
    shard_end = shard_start + shard_len
    # -----------------------------------------------------------------------

    ix = torch.randint(shard_start, shard_end, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(model, device, rank, world_size, num_batches=10):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch("val", BATCH_SIZE, BLOCK_SIZE, device, rank, world_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main():
    # FEAT 3/4: cada processo pega seu rank e sua própria GPU
    rank, local_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (rank == 0)

    config = Config({
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE,
        "dropout": DROPOUT,
    })

    model = ModeloCompleto(config).to(device)

    # FEAT 6: DDP no lugar de DataParallel
    model = DDP(model, device_ids=[local_rank])

    if is_main_process:
        parametros = int(sum(p.numel() for p in model.parameters()))
        print(f"Modelo: DDP | Parâmetros: {parametros:,}")
        print(f"World size: {world_size} | Batch local: {BATCH_SIZE} | "
              f"Batch efetivo global: {BATCH_SIZE * world_size}")
        print(f"{MAX_STEPS} steps | Grad Accum Steps: {GRAD_ACCUM_STEPS} | "
              f"LR: {LEARNING_RATE} | Min LR: {MIN_LR} | Warmup Steps: {WARMUP_STEPS}")
        print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(WEIGHT_DECAY),
    )

    model.train()
    step = 0
    val_loss = 0.0
    tokens_seen = 0  # <-- cumulativo, já contando todas as GPUs

    # tokens processados por step de otimização, somando todas as GPUs
    # (block_size * batch_size * grad_accum_steps é o valor por GPU;
    #  multiplicamos por world_size pois cada GPU faz essa mesma conta em paralelo)
    tokens_per_step = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS * world_size

    raw_model = model.module  # <-- para salvar, sempre acesse o modelo "puro" via .module

    while step < MAX_STEPS:
        # dt: sincroniza a GPU antes de marcar o tempo, pois chamadas CUDA são
        # assíncronas — sem isso o tempo medido seria só o de "disparar" os kernels,
        # não o de eles terminarem de fato.
        torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        step_loss_accum = 0.0

        lr = get_lr(step, LEARNING_RATE, WARMUP_STEPS, MAX_STEPS, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = get_batch("train", BATCH_SIZE, BLOCK_SIZE, device, rank, world_size)

            logits, loss = model(x, y)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss não finita no step {step}: {loss.item()}")

            step_loss_accum += loss.item()

            loss_scaled = loss / GRAD_ACCUM_STEPS
            loss_scaled.backward()
            # MUDANÇA 6 (cont.): o all-reduce dos gradientes entre GPUs acontece
            # automaticamente aqui dentro do .backward() do DDP.

        grad_norm = 0.0
        if GRAD_CLIP is not None:
            # A norma calculada aqui já reflete todas as GPUs: como o DDP fez
            # all-reduce (com média) dos gradientes durante o backward(), os
            # gradientes locais do rank 0 já são idênticos aos dos outros ranks
            # neste ponto — não é preciso nenhuma sincronização extra para a norma.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        # dt e throughput: de novo sincroniza antes de fechar o cronômetro,
        # para garantir que o step (forward+backward+optimizer) terminou de verdade.
        torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_seen += tokens_per_step
        tokens_per_sec = tokens_per_step / dt

        # MUDANÇA 8: eval, prints e save só no rank 0
        if is_main_process:
            if step % EVAL_INTERVAL == 0:
                val_loss = evaluate(model, device, rank, world_size)
                print(f"Validação | Step {step} | Val Loss: {val_loss:.4f} | LR: {lr:.10f}")

            if step % PRINT_INTERVAL == 0:
                print(
                    f"Step {step} | Loss: {step_loss_accum / GRAD_ACCUM_STEPS:.4f} | "
                    f"VAL Loss: {val_loss:.4f} | LR: {lr:.10f} | "
                    f"norm: {grad_norm:.4f} | dt: {dt*1000:.2f}ms | "
                    f"tok/s: {tokens_per_sec:,.0f} | tokens vistos: {tokens_seen:,}"
                )

        step += 1

    if is_main_process:
        print("Treinamento finalizado.")
        save_model(SAVE_DIR, raw_model)

    ddp_cleanup()


if __name__ == "__main__":
    main()
