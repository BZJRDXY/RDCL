import pytest
import torch

from rdcl.training.metrics import recall_and_ef_at_top_p, recall_at_top_p


def test_recall_and_ef_use_actual_selected_fraction_per_complex():
    logits = torch.tensor([
        10.0, 9.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
        10.0, 1.0, 0.0, -1.0,
    ])
    labels = torch.tensor([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ])
    edge_batch = torch.tensor([0] * 10 + [1] * 4)

    result = recall_and_ef_at_top_p(logits, labels, edge_batch, 2, top_p=0.15)

    assert result["recall"] == pytest.approx(1.0)
    # k/N is 2/10 for the first complex and 1/4 for the second.
    assert result["ef"] == pytest.approx((5.0 + 4.0) / 2.0)
    assert recall_at_top_p(logits, labels, edge_batch, 2, top_p=0.15) == pytest.approx(1.0)


def test_complexes_without_positive_labels_are_excluded():
    logits = torch.tensor([2.0, 1.0, 2.0, 1.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    edge_batch = torch.tensor([0, 0, 1, 1])

    result = recall_and_ef_at_top_p(logits, labels, edge_batch, 2, top_p=0.03)

    assert result == pytest.approx({"recall": 1.0, "ef": 2.0})


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1])
def test_invalid_top_p_is_rejected(top_p):
    with pytest.raises(ValueError):
        recall_and_ef_at_top_p(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([0]),
            1,
            top_p=top_p,
        )
