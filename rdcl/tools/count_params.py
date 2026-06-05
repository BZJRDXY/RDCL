from __future__ import annotations

import argparse

from rdcl.models.rdcl_model import RDCLBaseModel
from rdcl.utils import count_parameters, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = RDCLBaseModel(cfg)
    total, trainable = count_parameters(model)
    print(f"total={total:,}")
    print(f"trainable={trainable:,}")


if __name__ == "__main__":
    main()
