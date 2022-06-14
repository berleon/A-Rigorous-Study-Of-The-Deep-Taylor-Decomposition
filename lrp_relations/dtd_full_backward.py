#
# # Execute DTD-full-backward
#
# 1. Get samples from local linear segment (store as $X_L$).
# 2. For last layer, find which sample of $X_L$ would be a good root.
# 3. Recursively, derive and find other roots.


import dataclasses
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import savethat
import torch
import tqdm.auto

from lrp_relations import dtd, local_linear
from lrp_relations.utils import to_np


@dataclasses.dataclass(frozen=True)
class DTDFullbackwardArgs(savethat.Args):
    root_finder: str = "interpolation"
    rule: str = "z+"
    explained_output: int = 0
    n_points: int = 1_000


@dataclasses.dataclass(frozen=True)
class DTDFullbackwardResult:
    relevance_df: pd.DataFrame
    bbox_df: pd.DataFrame


class DTDFullbackward(
    savethat.Node[DTDFullbackwardArgs, DTDFullbackwardResult]
):
    def get_root_finder(self, mlp: dtd.MLP) -> dtd.InterpolationRootFinder:
        root_finder = dtd.InterpolationRootFinder(
            mlp,
            use_cache=True,
            use_candidates_cache=True,
            args=local_linear.InterpolationArgs(
                batch_size=50,
                n_refinement_steps=10,
                n_batches=1,
                show_progress=True,
                enforce_non_negative=True,
            ),
        )
        return root_finder

    def _run(self) -> DTDFullbackwardResult:
        explained_output_neuron = self.args.explained_output
        torch.manual_seed(1)
        mlp = dtd.MLP(3, 10, 10, 2)

        mlp.init_weights()

        explained_output: dtd.NEURON = slice(
            explained_output_neuron, explained_output_neuron + 1
        )

        x = mlp.get_input_with_output_greater(
            0.5, explained_output, non_negative=True
        )

        mlp_output = mlp.slice(output=explained_output)

        root_finder = self.get_root_finder(mlp_output)

        rel_fn_builder = dtd.FullBackwardFn.get_fn_builder(
            mlp_output,
            root_finder=root_finder,
            stabilize_grad=None,
        )
        rel_fns = dtd.get_decompose_relevance_fns(
            mlp_output, explained_output, rel_fn_builder
        )

        rel_fn = cast(dtd.FullBackwardFn, rel_fns[-1])

        dfs: list[pd.DataFrame] = []

        bbox = []
        for i in tqdm.auto.trange(self.args.n_points):
            x = mlp_output.get_input_with_output_greater(0.0, non_negative=True)
            rel = rel_fn(x)

            for idx, points in root_finder.candidates_cache.items():
                print(f"{idx} {len(points)}")
                vmax, _ = points.max(0)
                vmin, _ = points.min(0)

                bbox_dict = dataclasses.asdict(idx)
                bbox_dict.update(
                    dict(
                        input=to_np(x),
                        input_index=i,
                        vmin=to_np(vmin),
                        vmax=to_np(vmax),
                        bbox=to_np(vmax - vmin),
                    )
                )
                bbox.append(bbox_dict)

            root_finder.clear_caches()
            df = pd.DataFrame(rel.collect_info())
            df["network_input"] = [to_np(x)] * len(df)
            df["input_index"] = i
            dfs.append(df)

        return DTDFullbackwardResult(pd.concat(dfs), pd.DataFrame(bbox))

    def create_callgraph(self, rel: dtd.FullBackwardRel) -> None:
        import networkx as nx

        mlp = rel.computed_with_fn.mlp

        cg = nx.Graph()

        def visit_rel(rel: dtd.Relevance, prefix: str):
            rel_input = rel.computed_with_fn.get_input_layer()
            if isinstance(rel_input, dtd.LinearReLU):
                layer_name = "L" + str(mlp.get_layer_index(rel_input))
            else:
                layer_name = "o"

            node_name = prefix + layer_name
            cg.add_node(node_name)
            cg.add_edge(prefix, node_name)

            if isinstance(rel, dtd.FullBackwardRel):
                for root, rel_info in zip(
                    rel.roots, rel.relevance_upper_layers
                ):
                    j = root.explained_neuron
                    root_name = node_name + f"_r{j}@"

                    cg.add_node(root_name)
                    cg.add_edge(node_name, root_name)
                    visit_rel(rel_info, root_name)

        assert rel is not None
        visit_rel(rel, "s")
        cg.remove_node("s")

        plt.figure(figsize=(10, 10))

        pos = nx.nx_agraph.graphviz_layout(cg, prog="neato")
        nx.draw(
            cg,
            pos,
            with_labels=True,
            node_size=100,
            font_size=3,
            node_color="white",
        )

        plt.savefig(self.output_dir / "callgraph.svg")
