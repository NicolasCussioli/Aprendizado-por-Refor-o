"""Verifica completude e comparabilidade dos dados antes da entrega."""
import hashlib
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horta.experiments import METRICS, candidates, configuration


def main():
    config = configuration()
    seeds = [set(v) for v in config["seeds"].values()]
    assert sum(map(len, seeds)) == len(set.union(*seeds)), "Sobreposição de sementes entre etapas"
    winners = json.loads((ROOT / "results/winners.json").read_text(encoding="utf-8"))
    source = hashlib.sha256((ROOT / "horta/env.py").read_bytes()).hexdigest()
    total_steps, episode_count, checked_runs = 0, 0, 0
    tuning_rows = []
    for stage in ["pilot", "tuning", "final"]:
        choices = list(candidates(config))
        if stage == "pilot":
            choices = [key for key in choices if key[1] == "base"]
        elif stage == "final":
            choices = [key for key in choices if winners[key[0]] == key[1]]
        expected = {(a, c, s) for a, c in choices for s in config["seeds"][stage]}
        records = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "results" / stage).glob("*/run.json")]
        actual = {(r["algorithm"], r["config_id"], r["training_seed"]) for r in records}
        assert actual == expected, f"Execuções ausentes/extras em {stage}"
        assert len(records) == len(expected)
        eval_seeds = config["seeds"]["test" if stage == "final" else "validation"]
        for run in records:
            assert run["status"] == "completed"
            assert run["actual_steps"] == run["requested_steps"] == config["budgets"][stage]
            assert run["environment_sha256"] == source
            assert run["parameters_requested"] == candidates(config)[run["algorithm"], run["config_id"]]
            path = ROOT / f"results/{stage}/{run['algorithm']}__{run['config_id']}__{run['training_seed']}"
            episodes = pd.read_csv(path / "evaluation.csv")
            assert episodes.evaluation_seed.tolist() == eval_seeds
            assert np.isfinite(episodes.select_dtypes(include="number").to_numpy()).all()
            assert episodes.episode_length.between(1, 60).all()
            for metric in ["survival_fraction", "final_mean_health", "success"]:
                assert episodes[metric].between(0, 1).all()
            assert episodes.water_used.between(0, 60).all()
            for metric in METRICS:
                assert np.isclose(episodes[metric].mean(), run["evaluation_mean"][metric], atol=1e-10)
            monitor = pd.read_csv(path / "monitor.csv", skiprows=1)
            assert monitor.l.between(1, 60).all()
            assert 0 <= run["actual_steps"] - monitor.l.sum() < 60
            total_steps += run["actual_steps"]
            episode_count += len(episodes)
            checked_runs += 1
            if stage == "tuning":
                tuning_rows.append({"algorithm": run["algorithm"], "config_id": run["config_id"], **run["evaluation_mean"]})
    tuning = pd.DataFrame(tuning_rows)
    for algo in config["algorithms"]:
        means = tuning[tuning.algorithm == algo].groupby("config_id")[["episode_return", "water_used"]].mean()
        selected = means.sort_values(["episode_return", "water_used"], ascending=[False, True]).index[0]
        assert winners[algo] == selected, "Seleção não corresponde à validação"
    summary = pd.read_csv(ROOT / "results/final_summary.csv")
    for algo, group in pd.read_csv(ROOT / "results/final_runs.csv").groupby("algorithm"):
        row = summary[summary.algorithm == algo].iloc[0]
        assert row.training_seeds == 5
        for metric in METRICS:
            assert np.isclose(row[metric + "_mean"], group[metric].mean(), atol=1e-10)
            assert np.isclose(row[metric + "_std"], group[metric].std(ddof=1), atol=1e-10)
    for baseline in ["random", "heuristic"]:
        episodes = pd.read_csv(ROOT / f"results/{baseline}.csv")
        assert episodes.evaluation_seed.tolist() == config["seeds"]["test"]
        row = summary[summary.algorithm == baseline].iloc[0]
        for metric in METRICS:
            assert np.isclose(row[metric + "_mean"], episodes[metric].mean(), atol=1e-10)
            assert pd.isna(row[metric + "_std"])
        episode_count += len(episodes)
    manifest = json.loads((ROOT / "artifacts/models/manifest.json").read_text(encoding="utf-8"))
    for algo, item in manifest.items():
        assert item["training_seed"] == 101 and item["config_id"] == winners[algo]
        assert hashlib.sha256((ROOT / f"artifacts/models/{algo}_seed101.zip").read_bytes()).hexdigest() == item["sha256"]
    notebook = nbformat.read(ROOT / "trabalho_horta.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert not any(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
    assert (ROOT / "reports/relatorio_horta.html").stat().st_size > 1000
    assert checked_runs == 54 and total_steps == 8355840 and episode_count == 2870
    status = {"status": "passed", "training_runs": checked_runs, "training_steps": total_steps,
              "evaluated_episodes_including_baselines": episode_count, "winners": winners,
              "checks": ["completude", "sementes separadas", "orçamento", "hash do ambiente", "parâmetros",
                         "médias por episódio", "seleção por validação", "dispersão entre treinamentos",
                         "referências", "modelos publicados", "notebook sem erros", "relatório exportado"]}
    (ROOT / "results/verification.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
