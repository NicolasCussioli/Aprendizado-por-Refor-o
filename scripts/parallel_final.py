"""Agenda treinos finais independentes após terminar a busca de cada algoritmo.

Não compartilha modelos, dados ou episódios entre execuções. Máximo de cinco
processos de treino final; cada um usa uma thread de PyTorch.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(algo, config_id, seed, params, config):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from horta.experiments import run_trial
    return run_trial("final", algo, config_id, seed, params, config)


def main():
    from horta.experiments import configuration, candidates, collect_runs, select_winners
    config = configuration()
    params = candidates(config)
    scheduled = set()
    futures = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        while len(scheduled) < 3:
            runs = collect_runs("tuning")
            for algo in config["algorithms"]:
                if algo in scheduled or runs.empty:
                    continue
                completed = runs[runs.algorithm == algo]
                needed = len(config["algorithms"][algo]["variants"]) * len(config["seeds"]["tuning"])
                if len(completed) == needed:
                    chosen = select_winners(config, algorithm=algo)[algo]
                    print(f"Busca de {algo} concluída; final selecionado: {chosen}", flush=True)
                    for seed in config["seeds"]["final"]:
                        futures.append(pool.submit(worker, algo, chosen, seed, params[algo, chosen], config))
                    scheduled.add(algo)
            for future in futures:
                if future.done():
                    future.result()  # Propaga falhas, em vez de escondê-las.
            if len(scheduled) < 3:
                time.sleep(5)
        for future in futures:
            future.result()
    select_winners(config)
    collect_runs("final").to_csv(ROOT / "results/final_runs.csv", index=False)
    print("Todos os 15 treinamentos finais concluídos.", flush=True)


if __name__ == "__main__":
    main()
