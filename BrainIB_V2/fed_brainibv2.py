"""
Federated BrainIB V2 / SGSIB experiments for ABIDE and ADHD.

This script keeps the BrainIB V2 objective unchanged (classification loss on the
learned subgraph plus a matrix-Renyi mutual-information penalty between original
and subgraph embeddings) and adds a standard FedAvg outer loop where each site is
one client.

Example:
    python BrainIB_V2/fed_brainibv2.py
    python BrainIB_V2/fed_brainibv2.py --datasets abide --atlases AAL116 --rounds 50
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Reuse the original BrainIB V2 graph encoder and MI estimator.  The flexible
# subgraph generator below is the same MLP edge-mask idea as BrainIB V2, but it
# removes the original hard-coded 116-node assumption so harvard48 and
# schaefer100 can run without changing the original source files.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from SGSIB.GNN import GNN  # noqa: E402
from SGSIB.utils import calculate_MI  # noqa: E402


ABIDE_ALL_SITES = [
    "CALTECH",
    "KKI",
    "LEUVEN_1",
    "LEUVEN_2",
    "MAX_MUN",
    "NYU",
    "OLIN",
    "PITT",
    "SBL",
    "SDSU",
    "STANFORD",
    "TRINITY",
    "UCLA_1",
    "UCLA_2",
    "UM_1",
    "UM_2",
    "USM",
    "YALE",
]
ABIDE_TEST_SITES = ["NYU", "UM_1", "USM", "UCLA_1", "MAX_MUN", "PITT", "YALE", "KKI", "TRINITY", "STANFORD"]
ADHD_ALL_SITES = ["kki", "neuroimage", "nyu", "ohsu", "peking_1", "peking_2", "peking_3"]
ADHD_TEST_SITES = ["ohsu", "peking_1", "peking_2"]
ATLAS_NODE_COUNTS = {"harvard48": 48, "schaefer100": 100, "AAL116": 116}
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "auc"]


class FlexibleMLPSubgraph(nn.Module):
    """BrainIB V2-style MLP subgraph generator for arbitrary atlas sizes."""

    def __init__(self, node_features_num: int, edge_features_num: int, device: torch.device):
        super().__init__()
        self.device = device
        self.node_features_num = node_features_num
        self.edge_features_num = edge_features_num
        self.feature_size = 64
        self.linear = nn.Linear(self.node_features_num, self.feature_size).to(self.device)
        self.linear1 = nn.Linear(2 * self.feature_size, 8).to(self.device)
        self.linear2 = nn.Linear(8, 1).to(self.device)

    def _sample_graph(self, sampling_weights: torch.Tensor, temperature: float = 0.5, bias: float = 0.0) -> torch.Tensor:
        if self.training:
            bias = bias + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size(), device=sampling_weights.device) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            return torch.sigmoid(gate_inputs)
        return torch.sigmoid(sampling_weights)

    def _edge_prob_mat(self, graph: Data) -> torch.Tensor:
        graph = graph.to(self.device)
        node_count = graph.x.shape[0]
        x = self.linear(graph.x)
        f1 = x.unsqueeze(1).repeat(1, node_count, 1).view(-1, self.feature_size)
        f2 = x.unsqueeze(0).repeat(node_count, 1, 1).view(-1, self.feature_size)
        pair_features = torch.cat([f1, f2], dim=-1)
        mask_sigmoid = torch.sigmoid(self.linear2(torch.sigmoid(self.linear1(pair_features)))).reshape(node_count, node_count)
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[graph.edge_index[0], graph.edge_index[1]]
        return self._sample_graph(edge_mask, temperature=0.5, bias=0.0)

    def forward(self, graph: Data) -> Tuple[Data, torch.Tensor]:
        subgraph = graph.to(self.device)
        edge_mask = self._edge_prob_mat(subgraph)
        if subgraph.edge_attr is None:
            subgraph.edge_attr = edge_mask.reshape(-1, 1)
        else:
            subgraph.edge_attr = subgraph.edge_attr * edge_mask.reshape(-1, 1)
        pos_penalty = edge_mask.var()
        return subgraph, pos_penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated BrainIB V2 / SGSIB for ABIDE and ADHD")
    parser.add_argument("--abide_root", type=str, default=r"D:\Pycharm\Brain_Generalization_Projects\BrainGeneralization-main\data\preprocessed_data\abide")
    parser.add_argument("--adhd_root", type=str, default=r"D:\Pycharm\Brain_Generalization_Projects\BrainGeneralization-main\data\preprocessed_data\adhd_balanced_labels")
    parser.add_argument("--datasets", nargs="+", default=["abide", "adhd"], choices=["abide", "adhd"])
    parser.add_argument("--atlases", nargs="+", default=["harvard48", "schaefer100", "AAL116"], choices=list(ATLAS_NODE_COUNTS))
    parser.add_argument("--abide_test_sites", nargs="+", default=ABIDE_TEST_SITES)
    parser.add_argument("--adhd_test_sites", nargs="+", default=ADHD_TEST_SITES)
    parser.add_argument("--top_r", type=float, default=0.20, help="Fraction of strongest absolute correlations retained as edges.")
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--min_rounds", type=int, default=10, help="Best-round selection starts only after this many rounds.")
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_learning_rate", type=float, default=0.0005)
    parser.add_argument("--SGmodel_learning_rate", type=float, default=0.001)
    parser.add_argument("--mi_weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--results_root", type=str, default="fed_results")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def infer_label(sample_dir_name: str) -> int:
    if sample_dir_name.startswith("sub-control"):
        return 0
    if sample_dir_name.startswith("sub-patient"):
        return 1
    raise ValueError(f"Cannot infer label from sample folder name: {sample_dir_name}")


def infer_subject_id(sample_dir_name: str) -> str:
    for prefix in ("sub-control", "sub-patient"):
        if sample_dir_name.startswith(prefix):
            return sample_dir_name[len(prefix) :]
    raise ValueError(f"Cannot infer subject id from sample folder name: {sample_dir_name}")


def load_mat_square_matrix(mat_path: Path) -> np.ndarray:
    mat = scio.loadmat(mat_path)
    candidates = []
    for key, value in mat.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.ndim == 2 and value.shape[0] == value.shape[1] and np.issubdtype(value.dtype, np.number):
            candidates.append(value.astype(np.float32))
    if not candidates:
        raise ValueError(f"No square numeric matrix found in {mat_path}")
    matrix = max(candidates, key=lambda arr: arr.shape[0])
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def matrix_to_graph(matrix: np.ndarray, label: int, top_r: float) -> Data:
    node_count = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Correlation matrix must be square, got shape {matrix.shape}")
    if not 0 < top_r <= 1:
        raise ValueError("top_r must be in (0, 1].")

    upper_i, upper_j = np.triu_indices(node_count, k=1)
    scores = np.abs(matrix[upper_i, upper_j])
    edge_count = max(1, int(math.ceil(top_r * scores.size)))
    selected = np.argpartition(scores, -edge_count)[-edge_count:]
    src = upper_i[selected]
    dst = upper_j[selected]

    edge_index_np = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_weight_np = np.concatenate([matrix[src, dst], matrix[dst, src]]).astype(np.float32)

    return Data(
        x=torch.tensor(matrix, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight_np, dtype=torch.float32).reshape(-1, 1),
        y=torch.tensor([label], dtype=torch.long),
    )


def load_abide_site_map(abide_root: Path) -> Dict[str, str]:
    csv_path = abide_root / "Phenotypic_V1_0b.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ABIDE phenotypic CSV not found: {csv_path}")
    site_map: Dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sub_id = str(row["SUB_ID"]).strip()
            site_id = str(row["SITE_ID"]).strip()
            site_map[sub_id] = site_id
    return site_map


def load_dataset_by_site(dataset_name: str, atlas: str, root: Path, top_r: float) -> Dict[str, List[Data]]:
    data_by_site: Dict[str, List[Data]] = {}
    expected_nodes = ATLAS_NODE_COUNTS[atlas]

    if dataset_name == "abide":
        site_map = load_abide_site_map(root)
        sample_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(("sub-control", "sub-patient"))]
        for sample_dir in sorted(sample_dirs):
            subject_id = infer_subject_id(sample_dir.name)
            site = site_map.get(subject_id)
            if site is None:
                continue
            mat_path = sample_dir / f"{sample_dir.name}_{atlas}_correlation_matrix.mat"
            if not mat_path.exists():
                continue
            matrix = load_mat_square_matrix(mat_path)
            if matrix.shape[0] != expected_nodes:
                raise ValueError(f"{mat_path} has {matrix.shape[0]} nodes, expected {expected_nodes} for {atlas}.")
            data_by_site.setdefault(site, []).append(matrix_to_graph(matrix, infer_label(sample_dir.name), top_r))
    elif dataset_name == "adhd":
        for site_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
            site = site_dir.name
            for sample_dir in sorted([path for path in site_dir.iterdir() if path.is_dir() and path.name.startswith(("sub-control", "sub-patient"))]):
                mat_path = sample_dir / f"{sample_dir.name}_{atlas}_correlation_matrix.mat"
                if not mat_path.exists():
                    continue
                matrix = load_mat_square_matrix(mat_path)
                if matrix.shape[0] != expected_nodes:
                    raise ValueError(f"{mat_path} has {matrix.shape[0]} nodes, expected {expected_nodes} for {atlas}.")
                data_by_site.setdefault(site, []).append(matrix_to_graph(matrix, infer_label(sample_dir.name), top_r))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not data_by_site:
        raise RuntimeError(f"No samples loaded for dataset={dataset_name}, atlas={atlas}, root={root}")
    return data_by_site


def split_train_val(graphs: Sequence[Data], val_ratio: float, seed: int) -> Tuple[List[Data], List[Data]]:
    labels = np.array([int(graph.y.item()) for graph in graphs])
    indices = np.arange(len(graphs))
    stratify: Optional[np.ndarray] = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed, shuffle=True, stratify=stratify)
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]


def clone_models(global_model: GNN, global_sg_model: FlexibleMLPSubgraph, node_features: int, device: torch.device) -> Tuple[GNN, FlexibleMLPSubgraph]:
    model = GNN(num_of_features=node_features, device=device).to(device)
    sg_model = FlexibleMLPSubgraph(node_features_num=node_features, edge_features_num=1, device=device).to(device)
    model.load_state_dict(copy.deepcopy(global_model.state_dict()))
    sg_model.load_state_dict(copy.deepcopy(global_sg_model.state_dict()))
    return model, sg_model


def estimate_sigma(embeddings: torch.Tensor) -> float:
    z = embeddings.detach().cpu().numpy()
    if z.shape[0] <= 1:
        return 1.0
    diffs = z[:, None, :] - z[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
    distances = distances[~np.eye(distances.shape[0], dtype=bool)].reshape(distances.shape[0], -1)
    k = min(10, distances.shape[1])
    sigma = float(np.mean(np.sort(distances, axis=1)[:, :k]))
    return max(sigma, 1e-6)


def local_train(
    model: GNN,
    sg_model: FlexibleMLPSubgraph,
    train_graphs: Sequence[Data],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.train()
    sg_model.train()
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.model_learning_rate},
            {"params": sg_model.parameters(), "lr": args.SGmodel_learning_rate},
        ]
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0

    for _ in range(args.local_epochs):
        shuffled = list(train_graphs)
        random.shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            graphs = shuffled[start : start + args.batch_size]
            if len(graphs) < 2:
                continue
            batch_graph = next(iter(DataLoader(graphs, batch_size=len(graphs)))).to(device)
            embeddings, _ = model(batch_graph)

            subgraphs = []
            for graph in copy.deepcopy(graphs):
                subgraph, _ = sg_model(graph)
                subgraphs.append(subgraph.cpu())
            batch_subgraph = next(iter(DataLoader(subgraphs, batch_size=len(subgraphs)))).to(device)
            sub_embeddings, subgraph_output = model(batch_subgraph)

            sigma1 = estimate_sigma(embeddings)
            sigma2 = estimate_sigma(sub_embeddings)
            mi_loss = calculate_MI(embeddings, sub_embeddings, sigma1**2, sigma2**2) / len(graphs)
            labels = batch_graph.y.view(-1).to(device)
            classify_loss = criterion(subgraph_output, labels)
            loss = classify_loss + mi_loss * args.mi_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def fedavg_state_dicts(state_dicts: Sequence[MutableMapping[str, torch.Tensor]], weights: Sequence[float]) -> OrderedDict:
    averaged = OrderedDict()
    for key in state_dicts[0].keys():
        averaged[key] = sum(state[key].detach().cpu() * weight for state, weight in zip(state_dicts, weights))
    return averaged


def predict_probabilities(
    model: GNN,
    sg_model: FlexibleMLPSubgraph,
    graphs: Sequence[Data],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    sg_model.eval()
    labels_all: List[int] = []
    probs_all: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            graph_batch = graphs[start : start + batch_size]
            subgraphs = []
            for graph in copy.deepcopy(graph_batch):
                subgraph, _ = sg_model(graph)
                subgraphs.append(subgraph.cpu())
            batch = next(iter(DataLoader(subgraphs, batch_size=len(subgraphs)))).to(device)
            _, logits = model(batch)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            labels = batch.y.view(-1).detach().cpu().numpy().astype(int)
            labels_all.extend(labels.tolist())
            probs_all.extend(probs)
    return np.array(labels_all, dtype=int), np.array(probs_all, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {name: float("nan") for name in METRIC_NAMES}
    y_pred = np.argmax(probs, axis=1)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, probs[:, 1]) if len(np.unique(y_true)) == 2 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")
    return {key: float(value) for key, value in metrics.items()}


def evaluate_graphs(model: GNN, sg_model: FlexibleMLPSubgraph, graphs: Sequence[Data], args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    y_true, probs = predict_probabilities(model, sg_model, graphs, args.batch_size, device)
    return compute_metrics(y_true, probs)


def evaluate_clients(
    model: GNN,
    sg_model: FlexibleMLPSubgraph,
    client_graphs: Dict[str, Sequence[Data]],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    per_client = {site: evaluate_graphs(model, sg_model, graphs, args, device) for site, graphs in client_graphs.items() if len(graphs) > 0}
    pooled_graphs = [graph for graphs in client_graphs.values() for graph in graphs]
    pooled_metrics = evaluate_graphs(model, sg_model, pooled_graphs, args, device) if pooled_graphs else {name: float("nan") for name in METRIC_NAMES}
    return pooled_metrics, per_client


def macro_client_auc(per_client_metrics: Dict[str, Dict[str, float]]) -> float:
    auc_values = [metrics["auc"] for metrics in per_client_metrics.values() if not np.isnan(metrics["auc"])]
    return float(np.mean(auc_values)) if auc_values else float("nan")


def run_heldout_experiment(
    dataset_name: str,
    atlas: str,
    heldout_site: str,
    data_by_site: Dict[str, List[Data]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float | str | int]:
    if heldout_site not in data_by_site:
        raise ValueError(f"Held-out site {heldout_site} has no loaded samples for {dataset_name}-{atlas}.")

    node_features = ATLAS_NODE_COUNTS[atlas]
    train_by_site: Dict[str, List[Data]] = {}
    val_by_site: Dict[str, List[Data]] = {}
    for site, graphs in data_by_site.items():
        if site == heldout_site:
            continue
        train_graphs, val_graphs = split_train_val(graphs, args.val_ratio, args.seed)
        train_by_site[site] = train_graphs
        val_by_site[site] = val_graphs

    global_model = GNN(num_of_features=node_features, device=device).to(device)
    global_sg_model = FlexibleMLPSubgraph(node_features_num=node_features, edge_features_num=1, device=device).to(device)

    best_round = 0
    best_val_auc = -float("inf")
    best_model_state = copy.deepcopy(global_model.state_dict())
    best_sg_state = copy.deepcopy(global_sg_model.state_dict())
    rounds_without_improvement = 0

    for fed_round in range(1, args.rounds + 1):
        client_model_states = []
        client_sg_states = []
        client_weights = []
        local_losses = []
        total_train_samples = sum(len(graphs) for graphs in train_by_site.values())

        for site, graphs in train_by_site.items():
            if not graphs:
                continue
            client_model, client_sg_model = clone_models(global_model, global_sg_model, node_features, device)
            local_losses.append(local_train(client_model, client_sg_model, graphs, args, device))
            client_model_states.append(copy.deepcopy(client_model.state_dict()))
            client_sg_states.append(copy.deepcopy(client_sg_model.state_dict()))
            client_weights.append(len(graphs) / total_train_samples)

        global_model.load_state_dict(fedavg_state_dicts(client_model_states, client_weights))
        global_sg_model.load_state_dict(fedavg_state_dicts(client_sg_states, client_weights))
        global_model.to(device)
        global_sg_model.to(device)

        train_metrics, _ = evaluate_clients(global_model, global_sg_model, train_by_site, args, device)
        val_metrics, val_per_client = evaluate_clients(global_model, global_sg_model, val_by_site, args, device)
        val_macro_auc = macro_client_auc(val_per_client)
        test_metrics = evaluate_graphs(global_model, global_sg_model, data_by_site[heldout_site], args, device)

        print(
            f"[{dataset_name}-{atlas} heldout={heldout_site}] round={fed_round:03d} "
            f"loss={np.mean(local_losses):.5f} train_acc={train_metrics['accuracy']:.5f} "
            f"val_auc={val_metrics['auc']:.5f} val_macro_auc={val_macro_auc:.5f} test_auc={test_metrics['auc']:.5f}"
        )

        selectable = fed_round >= args.min_rounds and not np.isnan(val_macro_auc)
        if selectable and val_macro_auc > best_val_auc:
            best_val_auc = val_macro_auc
            best_round = fed_round
            best_model_state = copy.deepcopy(global_model.state_dict())
            best_sg_state = copy.deepcopy(global_sg_model.state_dict())
            rounds_without_improvement = 0
        elif fed_round >= args.min_rounds:
            rounds_without_improvement += 1

        if fed_round >= args.min_rounds and rounds_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at round {fed_round}; best round is {best_round}.")
            break

    global_model.load_state_dict(best_model_state)
    global_sg_model.load_state_dict(best_sg_state)
    global_model.to(device)
    global_sg_model.to(device)

    final_train_metrics, _ = evaluate_clients(global_model, global_sg_model, train_by_site, args, device)
    final_val_metrics, final_val_per_client = evaluate_clients(global_model, global_sg_model, val_by_site, args, device)
    final_test_metrics = evaluate_graphs(global_model, global_sg_model, data_by_site[heldout_site], args, device)

    row: Dict[str, float | str | int] = {
        "dataset": dataset_name,
        "atlas": atlas,
        "heldout_site": heldout_site,
        "best_round": best_round,
        "best_val_macro_auc": macro_client_auc(final_val_per_client),
        "train_samples": sum(len(graphs) for graphs in train_by_site.values()),
        "val_samples": sum(len(graphs) for graphs in val_by_site.values()),
        "test_samples": len(data_by_site[heldout_site]),
    }
    for split_name, metrics in (("train", final_train_metrics), ("val", final_val_metrics), ("test", final_test_metrics)):
        for metric_name, value in metrics.items():
            row[f"{split_name}_{metric_name}"] = value
    return row


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.5f}"
    return value


def write_results_csv(rows: List[Dict[str, object]], output_dir: Path) -> Path:
    metric_columns = [f"{split}_{metric}" for split in ("train", "val", "test") for metric in METRIC_NAMES]
    fieldnames = [
        "dataset",
        "atlas",
        "heldout_site",
        "best_round",
        "best_val_macro_auc",
        "train_samples",
        "val_samples",
        "test_samples",
        *metric_columns,
    ]

    summary_rows: List[Dict[str, object]] = []
    for summary_name, reducer in (("mean", np.nanmean), ("variance", np.nanvar), ("std", np.nanstd)):
        summary: Dict[str, object] = {
            "dataset": rows[0]["dataset"],
            "atlas": rows[0]["atlas"],
            "heldout_site": summary_name,
            "best_round": "",
            "best_val_macro_auc": reducer([float(row["best_val_macro_auc"]) for row in rows]),
            "train_samples": "",
            "val_samples": "",
            "test_samples": "",
        }
        for column in metric_columns:
            summary[column] = reducer([float(row[column]) for row in rows])
        summary_rows.append(summary)

    csv_path = output_dir / "test_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in [*rows, *summary_rows]:
            writer.writerow({key: format_csv_value(row.get(key, "")) for key in fieldnames})
    return csv_path


def run_dataset_atlas(dataset_name: str, atlas: str, args: argparse.Namespace, device: torch.device) -> Path:
    root = Path(args.abide_root if dataset_name == "abide" else args.adhd_root)
    test_sites = args.abide_test_sites if dataset_name == "abide" else args.adhd_test_sites
    data_by_site = load_dataset_by_site(dataset_name, atlas, root, args.top_r)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.results_root) / f"{dataset_name}_{atlas}_{timestamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = Path(args.results_root) / f"{dataset_name}_{atlas}_{timestamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for heldout_site in test_sites:
        rows.append(run_heldout_experiment(dataset_name, atlas, heldout_site, data_by_site, args, device))

    csv_path = write_results_csv(rows, output_dir)
    print(f"Saved {dataset_name}-{atlas} results to {csv_path}")
    return csv_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    for dataset_name in args.datasets:
        for atlas in args.atlases:
            run_dataset_atlas(dataset_name, atlas, args, device)


if __name__ == "__main__":
    main()
