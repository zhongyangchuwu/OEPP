from __future__ import annotations

import unittest

from oepp.api.common import API_CONFIG_ROOT, REPOSITORY_ROOT, load_yaml
from oepp.api.run import _prompt_path


class ApiConfigurationLocationTests(unittest.TestCase):
    def test_checked_in_configuration_stays_offline_and_resolves_its_prompt(self) -> None:
        config_path = API_CONFIG_ROOT / "siliconflow_qwen35_tablev_t4_3x3.yaml"
        config = load_yaml(config_path)
        self.assertFalse(config["experiment"]["api_enabled"])
        self.assertEqual(config["experiment"]["max_calls"], 0)
        _, prompt_path, mode = _prompt_path(config)
        self.assertEqual(mode, "table_v_legacy")
        self.assertTrue(prompt_path.is_file())
        self.assertTrue(prompt_path.is_relative_to(REPOSITORY_ROOT))

    def test_environment_template_is_separate_from_runtime_environment(self) -> None:
        template = REPOSITORY_ROOT / "configs" / "api" / ".env.example"
        self.assertTrue(template.is_file())
        self.assertEqual(template.name, ".env.example")


if __name__ == "__main__":
    unittest.main()
