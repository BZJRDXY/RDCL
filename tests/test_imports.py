def test_core_imports():
    import rdcl
    import rdcl.models.rdcl_model
    import rdcl.training.train_base
    import rdcl.inference.predict_from_cache
    import rdcl.tools.count_params


def test_model_class_available():
    from rdcl.models.rdcl_model import RDCLBaseModel
    assert RDCLBaseModel is not None
