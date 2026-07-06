import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tactile_vae import train


class TactileVAETrainConfigTest(unittest.TestCase):
    def test_yaml_config_populates_args_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                """
paths:
  data_root: /data/from_config
  output_dir: /tmp/tactile_vae_outputs
  run_name: config_run
data:
  data_format: parquet
  source_window: 32
  input_window: 8
  subsample_stride: 4
  stride: 8
  val_ratio: 0.1
  num_workers: 0
model:
  latent_dim: 64
  hidden_channels: 32
  bottleneck_channels: 64
  bottleneck_T: 2
  temporal_pool: flatten_mlp
  use_finger_embed: 0
loss:
  beta_kl: 0.002
  use_magnitude_weight: 0
train:
  epochs: 2
  batch_size: 16
  lr: 0.001
  max_steps: 7
  sample_latent: 0
logging:
  log_every: 3
  use_wandb: 1
  wandb_project: custom_project
  wandb_entity: berkeley_bair
""".strip()
            )

            args = train.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--batch_size",
                    "4",
                    "--use_wandb",
                    "0",
                    "--sample_latent",
                    "1",
                ]
            )

        self.assertEqual(args.data_root, "/data/from_config")
        self.assertEqual(args.output_dir, "/tmp/tactile_vae_outputs")
        self.assertEqual(args.run_name, "config_run")
        self.assertEqual(args.data_format, "parquet")
        self.assertEqual(args.source_window, 32)
        self.assertEqual(args.input_window, 8)
        self.assertEqual(args.stride, 8)
        self.assertEqual(args.val_ratio, 0.1)
        self.assertEqual(args.latent_dim, 64)
        self.assertEqual(args.temporal_pool, "flatten_mlp")
        self.assertEqual(args.use_finger_embed, 0)
        self.assertEqual(args.beta_kl, 0.002)
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.max_steps, 7)
        self.assertEqual(args.sample_latent, 1)
        self.assertEqual(args.use_wandb, 0)
        self.assertEqual(args.wandb_project, "custom_project")
        self.assertEqual(args.wandb_entity, "berkeley_bair")

    def test_wandb_init_uses_entity(self):
        args = train.parse_args(
            [
                "--data_root",
                "/data/root",
                "--output_dir",
                "/tmp/out",
                "--wandb_entity",
                "berkeley_bair",
            ]
        )
        cfg = train._build_config(args)

        fake_wandb = mock.Mock()
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            use_wandb = train._init_wandb(args, cfg, "entity_run")

        self.assertTrue(use_wandb)
        fake_wandb.init.assert_called_once()
        kwargs = fake_wandb.init.call_args.kwargs
        self.assertEqual(kwargs["project"], "trex_tactile_vae")
        self.assertEqual(kwargs["entity"], "berkeley_bair")
        self.assertEqual(kwargs["name"], "entity_run")


if __name__ == "__main__":
    unittest.main()
