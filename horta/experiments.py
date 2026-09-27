"""Execuções retomáveis: python -m horta.experiments --stage pilot|tuning|final."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from .env import HortaEnv, heuristic_action

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
METRICS = ["episode_return", "survival_fraction", "final_mean_health", "water_used", "success"]


def configuration():
    return json.loads((ROOT / "configs/experimentos.json").read_text(encoding="utf-8"))


def candidates(config=None):
    config = config or configuration()
    items = {}
    for algo, spec in config["algorithms"].items():
        for variant in spec["variants"]:
            params = copy.deepcopy(spec["base"])
            params.update(variant["changes"])
            items[algo, variant["id"]] = params
    return items


def evaluate(model, seeds, *, baseline=None):
    env = HortaEnv()
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        action_rng = np.random.default_rng(int(seed) + 100000)
        total = 0.0
        done = False
        while not done:
            if baseline == "random":
                action = int(action_rng.integers(6))
            elif baseline == "heuristic":
                action = heuristic_action(env)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        rows.append({"evaluation_seed": int(seed), "episode_return": total, "episode_length": env.turn,
                     "survival_fraction": info["survival_fraction"], "final_mean_health": info["final_mean_health"],
                     "water_used": info["water_used"], "success": int(info["is_success"])})
    env.close()
    return rows


def versions():
    return {p: importlib.metadata.version(p) for p in
            ["gymnasium", "stable-baselines3", "torch", "numpy", "pandas", "matplotlib", "pillow"]}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def run_trial(stage, algo, config_id, seed, params, config):
    trial_id = f"{algo}__{config_id}__{seed}"
    path = ROOT / "results" / stage / trial_id
    path.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256((ROOT / "horta/env.py").read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps({"params": params, "budget": config["budgets"][stage],
        "env": source_hash, "versions": versions(), "eval_seeds": config["seeds"]["test" if stage == "final" else "validation"]}, sort_keys=True).encode()).hexdigest()
    meta_path = path / "run.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise RuntimeError(f"Protocolo mudou para {trial_id}; use outra pasta/versão para preservar tentativas.")
        if old.get("status") == "completed":
            print(f"Já concluído: {stage}/{trial_id}", flush=True)
            return old
        archive = path / "retries" / f"attempt_{len(list((path / 'retries').glob('attempt_*'))) + 1}"
        archive.mkdir(parents=True, exist_ok=True)
        for artifact in path.iterdir():
            if artifact.is_file() and artifact.suffix != ".tmp":
                shutil.copy2(artifact, archive / artifact.name)
    meta = {"stage": stage, "algorithm": algo, "config_id": config_id, "training_seed": seed,
            "status": "running", "parameters_requested": params, "signature": signature,
            "environment_version": HortaEnv.VERSION, "environment_sha256": source_hash,
            "dependencies": versions(), "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                 "device": "cpu", "torch_threads": torch.get_num_threads()},
            "requested_steps": config["budgets"][stage]}
    atomic_json(meta_path, meta)
    model_dir = ROOT / "models" / stage
    model_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(HortaEnv(), filename=str(path / "monitor.csv"))
    start = time.perf_counter()
    print(f"Iniciando {stage}/{trial_id}: {meta['requested_steps']} passos", flush=True)
    try:
        model = ALGORITHMS[algo](config["policy"], env, seed=seed, device="cpu", verbose=0, **copy.deepcopy(params))
        model.learn(total_timesteps=meta["requested_steps"])
        meta["train_seconds"] = time.perf_counter() - start
        meta["actual_steps"] = int(model.num_timesteps)
        model_path = model_dir / trial_id
        model.save(model_path)
        meta["model_path"] = str(model_path.relative_to(ROOT)).replace("\\", "/") + ".zip"
        # Captura também os padrões efetivos da versão instalada, quando serializáveis.
        meta["parameters_effective"] = {key: value for key, value in vars(model).items()
              if isinstance(value, (str, int, float, bool, type(None))) and not key.startswith("_")}
        eval_seeds = config["seeds"]["test" if stage == "final" else "validation"]
        episodes = evaluate(model, eval_seeds)
        pd.DataFrame(episodes).to_csv(path / "evaluation.csv", index=False)
        meta["evaluation_mean"] = {m: float(np.mean([e[m] for e in episodes])) for m in METRICS}
        meta["status"] = "completed"
        atomic_json(meta_path, meta)
        print(f"Concluído {trial_id}: {meta['train_seconds']:.1f}s; retorno {meta['evaluation_mean']['episode_return']:.2f}", flush=True)
        return meta
    except Exception as exc:
        meta.update(status="failed", error=repr(exc), train_seconds=time.perf_counter() - start)
        atomic_json(meta_path, meta)
        raise
    finally:
        env.close()


def collect_runs(stage):
    rows = []
    for path in sorted((ROOT / "results" / stage).glob("*/run.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["status"] == "completed":
            rows.append({k: value[k] for k in ["algorithm", "config_id", "training_seed", "train_seconds", "actual_steps"]}
                        | value["evaluation_mean"])
    return pd.DataFrame(rows)


def select_winners(config, algorithm=None):
    runs = collect_runs("tuning")
    if algorithm and not runs.empty:
        runs = runs[runs.algorithm == algorithm]
    expected = {(a, c, s) for a, c in candidates(config) for s in config["seeds"]["tuning"] if algorithm is None or a == algorithm}
    actual = set(runs[["algorithm", "config_id", "training_seed"]].itertuples(index=False, name=None)) if not runs.empty else set()
    if actual != expected:
        raise RuntimeError(f"Busca incompleta: {len(actual)}/{len(expected)} execuções. Não iniciar teste.")
    summary = runs.groupby(["algorithm", "config_id"])[METRICS + ["train_seconds"]].agg(["mean", "std"])
    suffix = f"_{algorithm}" if algorithm else ""
    summary.to_csv(ROOT / f"results/tuning_summary{suffix}.csv")
    winner = {}
    for algo in ([algorithm] if algorithm else ALGORITHMS):
        ordered = runs[runs.algorithm == algo].groupby("config_id")[["episode_return", "water_used"]].mean()
        ordered = ordered.sort_values(["episode_return", "water_used"], ascending=[False, True])
        winner[algo] = str(ordered.index[0])
    atomic_json(ROOT / f"results/winners{suffix}.json", winner)
    return winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "tuning", "final", "baselines", "all"], default="all")
    parser.add_argument("--algorithm", choices=list(ALGORITHMS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = configuration()
    if config["status"] != "aprovado_conforme_usuario":
        raise RuntimeError("Registrar aprovação antes de iniciar.")
    choices = candidates(config)
    if args.algorithm:
        choices = {key: value for key, value in choices.items() if key[0] == args.algorithm}
    for stage in (["pilot", "tuning", "final", "baselines"] if args.stage == "all" else [args.stage]):
        if stage == "baselines":
            # Final deve existir para evitar uso prematuro de teste.
            final_runs = collect_runs("final")
            if len(final_runs) != 3 * len(config["seeds"]["final"]):
                raise RuntimeError("Concluir os treinamentos finais antes das referências no teste.")
            for baseline in ["random", "heuristic"]:
                rows = evaluate(None, config["seeds"]["test"], baseline=baseline)
                pd.DataFrame(rows).to_csv(ROOT / f"results/{baseline}.csv", index=False)
            continue
        selected = choices
        if stage == "pilot":
            selected = {(a, c): p for (a, c), p in choices.items() if c == "base"}
        elif stage == "final":
            winners = select_winners(config)
            selected = {(a, c): p for (a, c), p in choices.items() if winners[a] == c}
        for (algo, config_id), params in selected.items():
            for seed in config["seeds"][stage]:
                run_trial(stage, algo, config_id, seed, params, config)
        if not args.algorithm:
            collect_runs(stage).to_csv(ROOT / f"results/{stage}_runs.csv", index=False)
        if stage == "tuning":
            if not args.algorithm:
                print("Escolhas na validação:", select_winners(config), flush=True)


if __name__ == "__main__":
    main()
