import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check

from horta import HortaEnv


class EnvironmentTests(unittest.TestCase):
    def test_api(self):
        gym_check(HortaEnv(), skip_render_check=True)
        sb3_check(HortaEnv(), warn=True)

    def test_seed_and_weather_independent_of_actions(self):
        a, b = HortaEnv(), HortaEnv()
        np.testing.assert_array_equal(a.reset(seed=81)[0], b.reset(seed=81)[0])
        for _ in range(12):
            _, _, _, _, ia = a.step(5)
            _, _, _, _, ib = b.step(0)
            self.assertEqual(ia["weather"], ib["weather"])
            self.assertEqual(ia["rain"], ib["rain"])

    def test_water_conservation_and_refill(self):
        env = HortaEnv()
        env.reset(seed=3)
        env.water = 0.25
        _, _, _, _, info = env.step(0)
        self.assertEqual(env.water, 0)
        self.assertEqual(info["water_step"], 0.25)
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["water_step"], 0)
        self.assertEqual(info["invalid_action"], 1)
        env.step(4)
        self.assertEqual(env.water, env.CAPACITY)
        self.assertEqual(env.turn, 3)

    def test_death_is_irreversible(self):
        env = HortaEnv()
        env.reset(seed=10)
        env.health[0] = 0
        env.moisture[0] = 0.5
        env.step(0)
        self.assertEqual(env.health[0], 0)

    def test_terminal_horizon_and_bounds(self):
        env = HortaEnv()
        env.reset(seed=1)
        env.turn = 59
        obs, _, terminal, truncated, _ = env.step(5)
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertEqual(obs[-1], 0)
        self.assertTrue(env.observation_space.contains(obs))
        with self.assertRaises(RuntimeError):
            env.step(5)

    def test_render(self):
        env = HortaEnv(render_mode="rgb_array")
        env.reset(seed=1)
        self.assertEqual(env.render().shape, (600, 960, 3))
        self.assertEqual(env.render().dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
