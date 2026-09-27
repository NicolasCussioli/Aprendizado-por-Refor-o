"""Agrega resultados reais, gera figuras e grava execução ilustrativa."""
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .env import HortaEnv
from .experiments import ALGORITHMS, METRICS, ROOT, collect_runs, configuration, select_winners

COLORS = {"DQN": "#2976bb", "PPO": "#228657", "A2C": "#ca6a28", "random": "#777777", "heuristic": "#77549b"}


def summarize():
    config = configuration()
    winners = select_winners(config)
    final = collect_runs("final")
    expected = {(a, winners[a], s) for a in ALGORITHMS for s in config["seeds"]["final"]}
    actual = set(final[["algorithm", "config_id", "training_seed"]].itertuples(index=False, name=None))
    if actual != expected:
        raise RuntimeError("Treinos finais incompletos; não gerar conclusões finais.")
    rows = []
    for algo, group in final.groupby("algorithm"):
        row = {"algorithm": algo, "config_id": winners[algo], "training_seeds": len(group)}
        for metric in METRICS + ["train_seconds"]:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    for baseline in ["random", "heuristic"]:
        episodes = pd.read_csv(ROOT / f"results/{baseline}.csv")
        row = {"algorithm": baseline, "config_id": "fixed_rule", "training_seeds": 0}
        for metric in METRICS:
            row[metric + "_mean"] = float(episodes[metric].mean())
            row[metric + "_std"] = np.nan  # Episódios não são treinamentos independentes.
        row["train_seconds_mean"], row["train_seconds_std"] = 0.0, np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results/final_summary.csv", index=False)
    final.to_csv(ROOT / "results/final_runs.csv", index=False)
    collect_runs("tuning").to_csv(ROOT / "results/tuning_runs.csv", index=False)
    return summary, winners


def make_figures(summary, winners):
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = summary.algorithm.tolist()
    colors = [COLORS[a] for a in labels]
    for ax, metric, title in zip(axes, ["episode_return", "final_mean_health"], ["Retorno no teste", "Saúde final média"]):
        ax.bar(labels, summary[metric + "_mean"], color=colors, alpha=0.9,
               yerr=summary[metric + "_std"].fillna(0), capsize=5)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_ylim(0, 1.1)
    fig.suptitle("Média ± desvio entre 5 treinamentos; referências sem barras entre treinos", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "comparacao.png", dpi=160)
    plt.close(fig)

    config = configuration()
    grid = np.linspace(6000, config["budgets"]["final"], 100)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for algo, config_id in winners.items():
        curves = []
        for seed in config["seeds"]["final"]:
            monitor = pd.read_csv(ROOT / f"results/final/{algo}__{config_id}__{seed}/monitor.csv", skiprows=1)
            xs = monitor.l.cumsum().to_numpy()
            ys = monitor.r.rolling(100, min_periods=100).mean().to_numpy()
            mask = ~np.isnan(ys)
            curves.append(np.interp(grid, xs[mask], ys[mask], left=np.nan, right=ys[mask][-1]))
        matrix = np.array(curves)
        mean, std = np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0, ddof=1)
        ax.plot(grid, mean, label=algo, color=COLORS[algo])
        ax.fill_between(grid, mean - std, mean + std, color=COLORS[algo], alpha=0.15)
    ax.set(xlabel="Interações de treinamento", ylabel="Retorno de treino (média móvel de 100 episódios)",
           title="Configurações selecionadas: média e dispersão entre sementes")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "aprendizado.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in summary.iterrows():
        ax.scatter(row.water_used_mean, row.final_mean_health_mean, color=COLORS[row.algorithm], s=75)
        ax.annotate(row.algorithm, (row.water_used_mean, row.final_mean_health_mean), xytext=(5, 6), textcoords="offset points")
    ax.set(xlabel="Água aplicada por episódio (unidades sintéticas)", ylabel="Saúde final média", ylim=(0, 1.07),
           title="Consumo e saúde devem ser interpretados em conjunto")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "agua_saude.png", dpi=160)
    plt.close(fig)


def record_demo(winners):
    algo = "PPO"
    training_seed = configuration()["seeds"]["final"][0]
    evaluation_seed = configuration()["seeds"]["test"][0]
    model = ALGORITHMS[algo].load(ROOT / f"models/final/{algo}__{winners[algo]}__{training_seed}.zip", device="cpu")
    env = HortaEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=evaluation_seed)
    frames, trajectory = [env.render()], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminal, truncated, info = env.step(int(action.item()))
        frames.append(env.render())
        trajectory.append({"turn": env.turn, "action": int(action.item()), "reward": reward, **info})
        done = terminal or truncated
    (ROOT / "videos").mkdir(exist_ok=True)
    imageio.mimsave(ROOT / "videos/execucao_ppo.mp4", frames, fps=4, macro_block_size=1)
    imageio.mimsave(ROOT / "reports/figures/execucao_ppo.gif", [f[::2, ::2] for f in frames], duration=250, loop=0)
    pd.DataFrame(trajectory).to_csv(ROOT / "results/demo_trajectory.csv", index=False)
    (ROOT / "results/demo_selection.json").write_text(json.dumps({"algorithm": algo, "config_id": winners[algo],
        "training_seed": training_seed, "evaluation_seed": evaluation_seed,
        "reason": "PPO e primeiras sementes predefinidos; não representa a média de desempenho."}, indent=2, ensure_ascii=False), encoding="utf-8")
    env.close()


def main():
    summary, winners = summarize()
    make_figures(summary, winners)
    record_demo(winners)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
