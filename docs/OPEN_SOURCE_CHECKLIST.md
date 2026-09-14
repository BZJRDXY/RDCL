# Open-source Release Checklist

Before publishing:

- [ ] Remove private data paths from examples and logs.
- [ ] Do not commit `.pt`, `.pth`, or `.ckpt` weight files unless intentionally releasing them.
- [ ] Verify dependencies with `python -m rdcl.tools.check_dependencies --strict`.
- [x] Validate the released checkpoint with `python -m rdcl.tools.validate_checkpoint`.
- [ ] Run a cache-mode inference smoke test.
- [ ] Run a raw-structure inference smoke test if raw examples are available.
- [ ] Confirm the license and citation metadata.
- [ ] Confirm that `configs/rdcl_base_30m.yaml` contains the correct `inference.affinity_mean` and `inference.affinity_std` for the released weights.

Released checkpoint filename:

```text
checkpoints/rdcl_30m_state_dict.pt
```

This file is intentionally tracked with Git LFS. Other model checkpoints remain ignored by default.
