# from relation_network import dataset as rel_dataset
import pytest
import torch

from lrp_relations import data, gt_eval, utils

try:
    clevr_xai_available = utils.clevr_xai_path().exists()
except ValueError:
    clevr_xai_available = False


@pytest.mark.skipif(not clevr_xai_available, reason="CLEVR-XAI not available")
def test_load_clevr_xai_dataset():
    dataset = data.CLEVR_XAI()
    img, question, q_len, answer, idx = dataset[0]

    gt = dataset.get_ground_truth(idx)
    assert img.shape == (3, 128, 128)
    assert gt.shape == (128, 128)
    assert gt.dtype == torch.float

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=5,
        collate_fn=dataset.collate_data,
    )

    for imgs, questions, _, answers, indices in dataloader:
        gt_masks = torch.stack(
            [dataset.get_ground_truth(idx) for idx in indices]
        ).unsqueeze(1)
        assert imgs.shape == (5, 3, 128, 128)
        assert gt_masks.shape == (5, 1, 128, 128)
        assert gt_masks.dtype == torch.float

        assert answers.dtype == torch.int64
        assert answers.shape == (5,)
        assert len(questions) == 5
        break


def test_ration_in_heatmap():
    mask = torch.randn(128, 128) < 0
    heatmap = torch.randn(128, 128)
    ratio = gt_eval.get_ration_in_mask(
        heatmap.detach().numpy(), mask.detach().numpy()
    )
    assert ratio <= 0.75

    mask = torch.zeros(128, 128).bool()
    mask[0, 0] = 1
    heatmap = torch.randn(128, 128)
    heatmap[0, 0] = 1000000000.0
    ratio = gt_eval.get_ration_in_mask(
        heatmap.detach().numpy(), mask.detach().numpy()
    )
    assert ratio == 1.0
