import dataclasses
import json
import pickle
from collections import Counter
from pathlib import Path

import savethat
import torch
from savethat import logger
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from relation_network.dataset import CLEVR, collate_data, transform
from relation_network.model import RelationNetworks


@dataclasses.dataclass(frozen=True)
class TrainArgs(savethat.Args):
    data_root: str
    ckpt_path: str = "checkpoints"
    batch_size: int = 640
    lr: float = 5e-6
    lr_max: float = 5e-4
    lr_gamma = 2
    lr_step: int = 20
    clip_norm: float = 50
    reverse_question: bool = True
    weight_decay: float = 1e-4
    n_epoch: int = 260
    n_worker: int = 9
    data_parallel: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_and_decay: bool = False
    decay_lr_step: int = 20
    decay_lr_gamma: float = 0.5


def run_training(
    data_root: Path, ckpt_path: Path, args: TrainArgs
) -> dict[Path, float]:
    device = torch.device(args.device)

    def train(epoch):
        train_set = DataLoader(
            CLEVR(
                data_root,
                split="train",
                transform=transform,
                reverse_question=args.reverse_question,
                use_preprocessed=True,
            ),
            batch_size=args.batch_size,
            num_workers=args.n_worker,
            shuffle=True,
            collate_fn=collate_data,
        )

        dataset = iter(train_set)
        pbar = tqdm(dataset)
        moving_loss = 0

        relnet.train(True)
        for i, (image, question, q_len, answer, _) in enumerate(pbar):
            image, question, q_len, answer = (
                image.to(device),
                question.to(device),
                torch.tensor(q_len),
                answer.to(device),
            )

            relnet.zero_grad()
            output = relnet(image, question, q_len)
            loss = criterion(output, answer)
            loss.backward()
            nn.utils.clip_grad_norm_(relnet.parameters(), args.clip_norm)
            optimizer.step()

            output_max = output.data.cpu().numpy().argmax(1)
            correct = output_max == answer.data.cpu().numpy()
            correct = correct.sum() / args.batch_size

            if moving_loss == 0:
                moving_loss = correct

            else:
                moving_loss = moving_loss * 0.99 + correct * 0.01

            pbar.set_description(
                "Epoch: {}; Loss: {:.5f}; Acc: {:.5f}; LR: {:.6f}".format(
                    epoch + 1,
                    loss.detach().item(),
                    moving_loss,
                    optimizer.param_groups[0]["lr"],
                )
            )

    def valid(epoch: int) -> float:
        valid_set = DataLoader(
            CLEVR(
                data_root,
                "val",
                transform=None,
                reverse_question=args.reverse_question,
                use_preprocessed=True,
            ),
            batch_size=args.batch_size // 2,
            num_workers=4,
            collate_fn=collate_data,
        )
        dataset = iter(valid_set)

        relnet.eval()
        class_correct: Counter = Counter()
        class_total: Counter = Counter()

        with torch.no_grad():
            for image, question, q_len, answer, answer_class in tqdm(dataset):
                image, question, q_len = (
                    image.to(device),
                    question.to(device),
                    torch.tensor(q_len),
                )

                output = relnet(image, question, q_len)
                correct = output.data.cpu().numpy().argmax(1) == answer.numpy()
                for c, class_ in zip(correct, answer_class):
                    if c:
                        class_correct[class_] += 1
                    class_total[class_] += 1

        class_correct["total"] = sum(class_correct.values())
        class_total["total"] = sum(class_total.values())

        with open(ckpt_path / "log.jsonl", "a+") as w:
            info = {k: class_correct[k] / v for k, v in class_total.items()}
            info["epoch"] = epoch
            info["lr"] = optimizer.param_groups[0]["lr"]
            print(json.dumps(info), file=w)

        acc = class_correct["total"] / class_total["total"]
        print("Avg Acc: {:.5f}".format(acc))
        return acc

    def tidy_checkpoints(ckpts: dict[Path, float]) -> None:
        ckpts_sorted = sorted(ckpts.items(), key=lambda x: x[1])
        for ckpt, acc in ckpts_sorted[:-3]:
            if ckpt.exists():
                logger.debug(f"Removing {ckpt} with acc {acc}")
                ckpt.unlink()
            del ckpts[ckpt]

    def checkpoint(acc: float) -> None:
        ckpt_filename = ckpt_path / f"checkpoint_{str(epoch).zfill(3)}.model"
        ckpts[ckpt_filename] = acc
        logger.debug(f"Saving checkpoint to {ckpt_filename} with acc: {acc}")
        with open(ckpt_filename, "wb") as fb:
            torch.save(relnet.state_dict(), fb)

    with open(data_root / "dic.pkl", "rb") as f:
        dic = pickle.load(f)

    n_words = len(dic["word_dic"]) + 1

    relnet = RelationNetworks(n_words)
    if args.data_parallel:
        relnet = nn.DataParallel(relnet)
    relnet = relnet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        relnet.parameters(), lr=args.lr_max, weight_decay=args.weight_decay
    )
    scheduler: lr_scheduler._LRScheduler
    if args.warmup_and_decay:
        warm_up = lr_scheduler.LinearLR(
            optimizer, start_factor=args.lr / args.lr_max, total_iters=10
        )
        decay_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.decay_lr_step, gamma=args.decay_lr_gamma
        )
        scheduler = lr_scheduler.SequentialLR(
            schedulers=[warm_up, decay_scheduler], milestones=[100]
        )
    else:
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_gamma
        )

    ckpts: dict[Path, float] = {}

    with open(ckpt_path / "initial_weights.model", "wb") as fb:
        torch.save(relnet.state_dict(), fb)

    try:
        for epoch in range(args.n_epoch):
            train(epoch)
            scheduler.step()

            if scheduler.get_last_lr() < args.lr_max:  # type: ignore
                optimizer.param_groups[0]["lr"] = args.lr_max

            acc = valid(epoch)
            checkpoint(acc)
            tidy_checkpoints(ckpts)
    except Exception as e:
        logger.error(e)
    finally:
        return ckpts


@dataclasses.dataclass
class TrainedModel:
    checkpoints: dict[Path, float]


class Train(savethat.Node[TrainArgs, TrainedModel]):
    def _run(self):
        checkpoints = self.output_dir / "checkpoints"
        checkpoints.mkdir()

        ckpts = run_training(Path(self.args.data_root), checkpoints, self.args)

        ckpts_rel = {
            ckpt.relative_to(checkpoints): acc for ckpt, acc in ckpts.items()
        }

        return TrainedModel(ckpts_rel)
