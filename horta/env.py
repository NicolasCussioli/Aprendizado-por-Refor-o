"""MDP sintético de irrigação. Todas as unidades são adimensionais."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HortaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human", "ansi"], "render_fps": 4}
    VERSION = "1.0.0"
    HORIZON = 60
    CAPACITY = 8.0
    IRRIGATION = 0.20
    LOW = np.array([0.30, 0.38, 0.34, 0.42])
    HIGH = np.array([0.70, 0.75, 0.68, 0.80])
    DEMAND = np.array([0.8, 1.0, 1.2, 1.1])
    DRAINAGE = np.array([0.010, 0.012, 0.016, 0.009])
    WEATHER = ("Seco", "Ameno", "Chuvoso")
    TRANSITION = np.array([[0.80, 0.18, 0.02], [0.25, 0.55, 0.20], [0.10, 0.30, 0.60]])
    EVAPORATION = np.array([0.018, 0.012, 0.006])
    RAIN_PROBABILITY = np.array([0.05, 0.15, 0.65])
    RAIN_AMOUNT = np.array([0.02, 0.04, 0.08])
    ACTIONS = ("Irrigar 1", "Irrigar 2", "Irrigar 3", "Irrigar 4", "Reabastecer", "Esperar")

    def __init__(self, render_mode=None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode inválido: {render_mode}")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(13,), dtype=np.float32)
        self._window = None
        self._clock = None
        self._done = True

    def _obs(self):
        return np.concatenate((self.moisture, self.health, [self.water / self.CAPACITY],
                               np.eye(3)[self.weather], [(self.HORIZON - self.turn) / self.HORIZON])).astype(np.float32)

    def _info(self):
        return {"water_used": float(self.water_used), "survival_fraction": float(np.mean(self.health > 0)),
                "final_mean_health": float(self.health.mean()), "weather": int(self.weather),
                "turn": int(self.turn), "is_success": bool(self.turn == self.HORIZON and
                       np.all(self.health > 0) and self.health.mean() >= 0.5)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.moisture = self.np_random.uniform(0.40, 0.70, size=4)
        self.health = np.ones(4, dtype=np.float64)
        self.water = float(self.np_random.uniform(4, self.CAPACITY))
        self.weather = int(self.np_random.choice(3, p=[0.50, 0.35, 0.15]))
        self.turn = 0
        self.water_used = 0.0
        self.last_action = "Início"
        self._done = False
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("Execute reset antes de iniciar ou continuar um episódio encerrado.")
        if not self.action_space.contains(action):
            raise ValueError(f"Ação inválida: {action}")
        action = int(action)
        alive = self.health > 0
        used, refill, invalid = 0.0, 0, 0
        if action < 4:
            if alive[action] and self.water > 0:
                used = min(1.0, self.water)
                self.moisture[action] += self.IRRIGATION * used
                self.water -= used
            else:
                invalid = 1
        elif action == 4:
            self.water = self.CAPACITY
            refill = 1

        # Número/ordem de sorteios são independentes da ação: clima pareado por semente.
        rain = self.RAIN_AMOUNT[self.weather] if self.np_random.random() < self.RAIN_PROBABILITY[self.weather] else 0.0
        evaporation = self.EVAPORATION[self.weather] * self.DEMAND * self.np_random.uniform(0.9, 1.1, 4)
        self.moisture = np.clip(self.moisture + rain - evaporation - self.DRAINAGE, 0, 1)
        dry = np.maximum(self.LOW - self.moisture, 0)
        wet = np.maximum(self.moisture - self.HIGH, 0)
        in_band = (dry == 0) & (wet == 0)
        delta = np.where(in_band, 0.015, np.where(dry > 0, -0.04 - 0.30 * dry, -0.02 - 0.15 * wet))
        self.health = np.where(alive, np.clip(self.health + delta, 0, 1), 0)
        deaths = int(np.sum(alive & (self.health == 0)))
        reward = float(self.health.mean() - 0.10 * used - 2.0 * deaths - 0.05 * refill - 0.10 * invalid)
        self.water_used += used
        self.weather = int(self.np_random.choice(3, p=self.TRANSITION[self.weather]))
        self.turn += 1
        self.last_action = self.ACTIONS[action]
        self._done = self.turn >= self.HORIZON or not np.any(self.health > 0)
        info = self._info()
        info.update({"rain": float(rain), "water_step": used, "new_deaths": deaths, "invalid_action": invalid})
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, bool(self._done), False, info

    def render(self):
        if self.render_mode == "ansi":
            return (f"Turno {self.turn}/{self.HORIZON} | {self.WEATHER[self.weather]} | água {self.water:.1f}/8\n"
                    f"Umidade {np.round(self.moisture, 2)} | Saúde {np.round(self.health, 2)} | {self.last_action}")
        if self.render_mode is None:
            return None
        from PIL import Image, ImageDraw, ImageFont
        canvas = Image.new("RGB", (960, 600), "#f5f3e9")
        draw = ImageDraw.Draw(canvas)
        from matplotlib.font_manager import findfont
        font_path = findfont("DejaVu Sans")
        title = ImageFont.truetype(font_path, 26)
        font = ImageFont.truetype(font_path, 18)
        draw.text((30, 20), "Horta comunitária", font=title, fill="#243a32")
        draw.text((30, 60), f"Turno {self.turn}/60   |   Clima: {self.WEATHER[self.weather]}   |   {self.last_action}", font=font, fill="#243a32")
        for i in range(4):
            x, y = 30 + (i % 2) * 330, 120 + (i // 2) * 220
            draw.rounded_rectangle((x, y, x + 300, y + 195), radius=12, fill="#ffffff", outline="#a8b5a1", width=2)
            draw.text((x + 15, y + 12), f"Canteiro {i + 1}", font=font, fill="#243a32")
            leaf = "#3d8c55" if self.health[i] > 0 else "#827569"
            draw.line((x + 270, y + 75, x + 270, y + 150), fill=leaf, width=5)
            draw.ellipse((x + 240, y + 72, x + 270, y + 105), fill=leaf)
            draw.ellipse((x + 270, y + 93, x + 295, y + 126), fill=leaf)
            for j, (label, value, color) in enumerate((("Umidade", self.moisture[i], "#468dcc"), ("Saúde", self.health[i], leaf))):
                by = y + 60 + 60 * j
                draw.text((x + 15, by - 5), f"{label}: {value:.0%}", font=font, fill="#243a32")
                draw.rectangle((x + 15, by + 25, x + 225, by + 41), fill="#e4e8e2")
                draw.rectangle((x + 15, by + 25, x + 15 + int(210 * value), by + 41), fill=color)
                if j == 0:
                    for boundary in (self.LOW[i], self.HIGH[i]):
                        bx = x + 15 + int(210 * boundary)
                        draw.line((bx, by + 21, bx, by + 45), fill="#334333", width=2)
        draw.text((730, 120), "Reservatório", font=font, fill="#243a32")
        draw.rectangle((765, 170, 885, 430), fill="#d9e4eb", outline="#46677d", width=3)
        draw.rectangle((768, 427 - int(254 * self.water / 8), 882, 427), fill="#468dcc")
        draw.text((740, 450), f"{self.water:.1f}/8 unidades", font=font, fill="#243a32")
        draw.text((30, 565), f"Água usada: {self.water_used:.1f}   |   Simulação didática; unidades sintéticas", font=font, fill="#243a32")
        frame = np.asarray(canvas).copy()
        if self.render_mode == "rgb_array":
            return frame
        import pygame
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode((960, 600))
            self._clock = pygame.time.Clock()
        pygame.event.pump()
        self._window.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._window = None


def heuristic_action(env: HortaEnv):
    """Irriga o maior déficit; reabastece se houver necessidade sem água."""
    deficit = np.where(env.health > 0, env.LOW + 0.07 - env.moisture, -np.inf)
    if np.max(deficit) <= 0:
        return 5
    if env.water < 1:
        return 4
    return int(np.argmax(deficit))
