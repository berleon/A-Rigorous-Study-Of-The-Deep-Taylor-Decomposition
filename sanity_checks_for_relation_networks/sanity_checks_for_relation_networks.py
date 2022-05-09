"""Main module."""

import argparse
import dataclasses

import savethat


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name", type=str, help="folder in which to store results"
    )

    parser.add_argument(
        "--epochs", type=int, default=1, help="epochs to train."
    )
    parser.add_argument(
        "--hidden_dims_g",
        nargs="+",
        type=int,
        default=[256, 256, 256],
        help="layers of relation function g",
    )

    parser.add_argument(
        "--output_dim_g",
        type=int,
        default=256,
        help="output dimension of relation function g",
    )
    parser.add_argument(
        "--hidden_dims_f",
        nargs="+",
        type=int,
        default=[256, 512],
        help="layers of final network f",
    )
    parser.add_argument(
        "--hidden_dim_lstm", type=int, default=32, help="units of LSTM"
    )
    parser.add_argument(
        "--lstm_layers", type=int, default=1, help="layers of LSTM"
    )

    parser.add_argument(
        "--emb_dim", type=int, default=32, help="word embedding dimension"
    )
    parser.add_argument("--batch_size", type=int, default=3, help="batch size")

    parser.add_argument("--dropout", action="store_true", help="enable dropout")
    parser.add_argument(
        "--tanh_act",
        action="store_true",
        help="use tanh activation for MLP instead of relu",
    )
    parser.add_argument(
        "--wave_penc",
        action="store_true",
        help="use sin/cos positional encoding instead of one-of-k",
    )

    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    parser.add_argument(
        "--babi_tasks",
        nargs="+",
        type=int,
        default=-1,
        help="which babi task to train and test. -1 to select all of them.",
    )

    parser.add_argument(
        "--split_manually",
        action="store_true",
        help="Use en-10k folder instead of en-valid-10k folder of babi. "
        "Active only with --babi_tasks specified.",
    )
    parser.add_argument(
        "--only_relevant",
        action="store_true",
        help="read only relevant fact from babi dataset. "
        "Active only with --split_manually",
    )

    # optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="optimizer hyperparameter"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="optimizer hyperparameter",
    )

    parser.add_argument(
        "--test_on_test",
        action="store_true",
        help="final test on test set instead of validation set",
    )
    parser.add_argument(
        "--test_jointly", action="store_true", help="final test on all tasks"
    )
    parser.add_argument("--cuda", action="store_true", help="use gpu")
    parser.add_argument("--load", action="store_true", help=" load saved model")
    parser.add_argument(
        "--no_save", action="store_true", help="disable model saving"
    )

    return parser


@dataclasses.dataclass(frozen=True)
class SanityChecksForRelationNetworksArgs(savethat.Args):
    pass


class SanityChecksForRelationNetworks(savethat.Node):
    def _run(self):
        parser = get_parser()
        args = parser.parse_args()
        print(args)
