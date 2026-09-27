"""Executa um modelo publicado: python -m horta.demo --algorithm PPO --mode human."""
import argparse

from .env import HortaEnv
from .experiments import ALGORITHMS, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="PPO")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--mode", choices=["human", "ansi"], default="ansi")
    args = parser.parse_args()
    model = ALGORITHMS[args.algorithm].load(ROOT / f"artifacts/models/{args.algorithm}_seed101.zip", device="cpu")
    env = HortaEnv(render_mode=args.mode)
    obs, _ = env.reset(seed=args.seed)
    total, done = 0.0, False
    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            total += reward
            done = terminated or truncated
            if args.mode == "ansi":
                print(env.render())
        print(f"Retorno: {total:.2f}; saúde: {info['final_mean_health']:.3f}; água: {info['water_used']:.1f}; sucesso: {info['is_success']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
