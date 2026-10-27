"""Train a StructTokenBench VQ-VAE tokenizer (AminoAseed / ESM3-style) on our corpus.

What it does
------------
Runs StructTokenBench's own trainer code unmodified on the BioEmu-CATH + MISATO
ensemble frames. Three compatibility shims replace DeepSpeed with DDP:
  1. deepspeed stubbed (upstream imports it at module scope)
  2. DDP strategy gets a .config dict (upstream writes to it in setup())
  3. model_init_fn replaced with a plain constructor call

Paper
-----
`appendix.tex` static retrain appendix (promises P-17, P-26, P-42):
"every static tokenizer, retrained on exactly our data with exactly our
splits and budget".

Origin
------
Copied from submission_exp/src/rebuttal_analyses/sametrain_baselines/
train_sametrain.py. Changed: paths resolved through ensembits.paths,
manifest/extras_dir injected into config, docstring rewritten, provenance
header added. Training logic unchanged.

How to run
----------
  python train_static.py --config config/aminoaseed_retrained.yaml \\
      --manifest /path/to/retrain_manifest.json \\
      --extras-dir /path/to/bioemu_cath_extras \\
      --output-dir /path/to/output/static_retrain/aminoaseed_retrained \\
      --devices 0
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
STB_SRC = os.environ.get(
    "ENSEMBITS_STB_SRC",
    os.path.join(HERE, "..", "..", "data", "external", "StructTokenBench", "src"),
)
if not os.path.isdir(STB_SRC):
    STB_SRC = os.environ.get("STB_SRC", "")


def stub_deepspeed():
    if "deepspeed" in sys.modules:
        return
    ds = types.ModuleType("deepspeed")
    ds.utils = types.ModuleType("deepspeed.utils")
    ds.utils.safe_get_full_grad = lambda p: p.grad
    ds.zero = types.ModuleType("deepspeed.zero")

    class _NoopInit:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return None
        def __exit__(self, *a):
            return False

    ds.zero.Init = _NoopInit
    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.utils"] = ds.utils
    sys.modules["deepspeed.zero"] = ds.zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True,
                    help="path to retrain_manifest.json")
    ap.add_argument("--extras-dir", default="",
                    help="path to bioemu_cath_extras/ (for BioEmu sequences)")
    ap.add_argument("--output-dir", required=True,
                    help="training output directory")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--devices", default="0",
                    help="GPU device(s), e.g. '0' or '0,1'")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--fast-train-metrics", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    stub_deepspeed()
    sys.path.insert(0, STB_SRC)
    sys.path.insert(0, os.path.join(STB_SRC, "baselines"))
    sys.path.insert(0, HERE)

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.loggers import CSVLogger
    from omegaconf import OmegaConf

    import vqvae_model
    from torch.utils.data import DataLoader
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    from ensemble_frames_dataset import EnsembleFramesDataset
    from ema_quantizer import EMAQuantizer

    vqvae_model.EMAQuantizer = EMAQuantizer

    class OurPretrainDataModule(pl.LightningDataModule):
        def __init__(self, cfg, py_logger):
            super().__init__()
            self.cfg = cfg
            self.py_logger = py_logger
            self.all_split_names = ["validation"]

        def _build(self, split):
            kwargs = OmegaConf.to_container(self.cfg.data, resolve=True)
            kwargs.update({"split": split, "py_logger": self.py_logger,
                           "seq_tokenizer": EsmSequenceTokenizer(),
                           "in_memory": False})
            return EnsembleFramesDataset(**kwargs)

        def _loader_kwargs(self):
            nw = self.cfg.data.num_workers
            kw = {"batch_size": self.cfg.optimization.micro_batch_size,
                  "num_workers": nw, "pin_memory": True}
            if nw > 0:
                kw["prefetch_factor"] = self.cfg.data.prefetch_factor
            return kw

        def train_dataloader(self):
            if not hasattr(self, "train_hf_dataset"):
                self.train_hf_dataset = self._build("train")
            ds = self.train_hf_dataset
            self.py_logger.info(f"train samples: {len(ds)}")
            return DataLoader(ds, collate_fn=ds.collate_fn, shuffle=True,
                              drop_last=True, **self._loader_kwargs())

        def val_dataloader(self):
            loaders = []
            for split in self.all_split_names:
                if not hasattr(self, f"{split}_hf_dataset"):
                    setattr(self, f"{split}_hf_dataset", self._build(split))
                ds = getattr(self, f"{split}_hf_dataset")
                self.py_logger.info(f"{split} samples: {len(ds)}")
                loaders.append(DataLoader(ds, collate_fn=ds.collate_fn,
                                          shuffle=False, **self._loader_kwargs()))
            return loaders

        def setup(self, stage=None):
            pass

    def model_init_fn(trainer, model_cfg, **model_kwargs):
        return vqvae_model.VQVAEModel(model_cfg, **model_kwargs)

    vqvae_model.model_init_fn = model_init_fn

    skip_metrics = {"on": False}
    if args.fast_train_metrics:
        import numpy as np

        class _SkippedChain:
            def __getitem__(self, _):
                return self
            def rmsd(self, *a, **k):
                return 0.0
            def lddt_ca(self, *a, **k):
                return np.array([0.0])

        _real_from_bb = vqvae_model.WrappedProteinChain.from_backbone_atom_coordinates
        def _maybe_from_bb(*a, **k):
            return _SkippedChain() if skip_metrics["on"] else _real_from_bb(*a, **k)
        vqvae_model.WrappedProteinChain.from_backbone_atom_coordinates = staticmethod(
            _maybe_from_bb)

    cfg = OmegaConf.load(args.config)

    # Inject runtime paths
    cfg.data.manifest_path = args.manifest
    cfg.data.extras_dir = args.extras_dir
    cfg.trainer.default_root_dir = args.output_dir
    cfg.trainer.devices = [int(d) for d in args.devices.split(",")]

    if args.smoke:
        cfg.data.fast_dev_run = True
        cfg.data.num_workers = 0
        cfg.trainer.devices = cfg.trainer.devices[:1]
        cfg.trainer.max_steps = cfg.max_steps = 20
        cfg.trainer.limit_val_batches = 4
        cfg.trainer.val_check_interval = 10
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.enable_model_summary = True
        cfg.optimization.scheduler.num_warmup_steps = 5
        cfg.optimization.scheduler.num_training_steps = 20
        cfg.experiment_name = cfg.experiment_name + "_smoke"

    if args.max_steps:
        cfg.trainer.max_steps = cfg.max_steps = args.max_steps
        cfg.optimization.scheduler.num_training_steps = args.max_steps

    root = cfg.trainer.default_root_dir
    os.makedirs(root, exist_ok=True)
    cfg.save_dir_path = os.path.join(root, f"{cfg.experiment_name}_final.ckpt")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    logger = logging.getLogger(cfg.experiment_name)
    logger.info(f"config: {args.config}")
    logger.info(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.optimization.seed)

    strategy = DDPStrategy(find_unused_parameters=True)
    strategy.config = {"zero_optimization": {"stage": 2},
                       "train_micro_batch_size_per_gpu": cfg.optimization.micro_batch_size}

    class EpochSeedCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            ds = getattr(trainer.datamodule, "train_hf_dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(trainer.current_epoch)

    class TrainMetricGate(pl.Callback):
        def on_train_batch_start(self, *a, **k):
            skip_metrics["on"] = args.fast_train_metrics
        def on_validation_start(self, *a, **k):
            skip_metrics["on"] = False

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(root, "checkpoints"),
            monitor=cfg.checkpoint.monitor, mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
            filename="{epoch}-{step}-{validation_bb_rmsd:.3f}",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.progress.TQDMProgressBar(refresh_rate=50),
        EpochSeedCallback(),
        TrainMetricGate(),
    ]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=callbacks,
        plugins=[],
        strategy=strategy,
        logger=CSVLogger(save_dir=root, name=cfg.experiment_name),
    )

    datamodule = OurPretrainDataModule(cfg, logger)
    datamodule.setup()

    model = vqvae_model.LightningVQPretrainModel(
        model_cfg=cfg.model,
        trainer=trainer,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        all_split_names=datamodule.all_split_names,
    )

    resume_from = args.resume or (cfg.model.ckpt_path or None)
    logger.info(f"*********** start training (resume={resume_from}) ***********")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)
    logger.info(f"saved final weights -> {cfg.save_dir_path}")


if __name__ == "__main__":
    main()
