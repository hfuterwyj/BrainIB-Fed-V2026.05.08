"""
Federated BrainIBV2/SGSIB runner for ABIDE and ADHD site-held-out studies.

This script keeps the BrainIBV2 model idea intact (GCN backbone + SOPOOL + MLP
classifier, trained together with the MLP subgraph generator and the same
matrix-entropy mutual-information objective) while adding only the federated
FedAvg orchestration, ABIDE/ADHD loaders, site-wise validation, and CSV result
export required for the federated experiments.
"""

import argparse
import copy
import csv
import math
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


ABIDE_SITES = [
    "CALTECH", "KKI", "LEUVEN_1", "LEUVEN_2", "MAX_MUN", "NYU", "OLIN", "PITT", "SBL",
    "SDSU", "STANFORD", "TRINITY", "UCLA_1", "UCLA_2", "UM_1", "UM_2", "USM", "YALE",
]
ABIDE_TEST_SITES = ["NYU", "UM_1", "USM", "UCLA_1", "MAX_MUN", "PITT", "YALE", "KKI", "TRINITY", "STANFORD"]
ADHD_SITES = ["kki", "neuroimage", "nyu", "ohsu", "peking_1", "peking_2", "peking_3"]
ADHD_TEST_SITES = ["ohsu", "peking_1", "peking_2"]
PARCELLATIONS = {"harvard48": 48, "schaefer100": 100, "AAL116": 116}
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "auc"]


class FedBrainIBGNN(nn.Module):
    """BrainIBV2 GNN backbone with dynamic node-feature dimensions for all atlases."""

    def __init__(self, num_of_features: int, device: torch.device):
        super().__init__()
        self.device = device
        self.graph_conv_1 = GCNConv(num_of_features, 128)
        self.graph_conv_2 = GCNConv(128, 128)
        self.SOPOOL = nn.Sequential(OrderedDict([
            ("Linear_1", nn.Linear(128, 32)),
            ("ReLU_1", nn.ReLU()),
            ("Linear_2", nn.Linear(32, 32)),
            ("ReLU_2", nn.ReLU()),
            ("Linear_3", nn.Linear(32, 32)),
            ("ReLU_3", nn.ReLU()),
        ]))
        self.MLP_1 = nn.Sequential(OrderedDict([
            ("Linear_1", nn.Linear(32 ** 2, 32)),
            ("ReLU_1", nn.ReLU()),
            ("Linear_2", nn.Linear(32, 32)),
            ("ReLU_2", nn.ReLU()),
            ("Linear_3", nn.Linear(32, 2)),
            ("ReLU_3", nn.ReLU()),
        ]))

    def forward(self, graph_batch: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_batch = graph_batch.to(self.device)
        edge_weight = graph_batch.edge_attr.view(-1)
        node_features_1 = F.relu(self.graph_conv_1(graph_batch.x, graph_batch.edge_index, edge_weight=edge_weight))
        node_features_2 = F.relu(self.graph_conv_2(node_features_1, graph_batch.edge_index, edge_weight=edge_weight))
        node_features_ = F.dropout(node_features_2, p=0.5, training=self.training)
        normalized_node_features = F.normalize(node_features_, dim=1)

        hh_rows = []
        for graph_idx in range(len(graph_batch.ptr) - 1):
            start = int(graph_batch.ptr[graph_idx])
            end = int(graph_batch.ptr[graph_idx + 1])
            graph = self.SOPOOL(normalized_node_features[start:end])
            hh_rows.append(torch.mm(graph.t(), graph).view(1, -1))
        hh_tensor = torch.cat(hh_rows, dim=0)
        output = F.dropout(self.MLP_1(hh_tensor), p=0.5, training=self.training)
        return hh_tensor, output


class FedMLPSubgraph(nn.Module):
    """BrainIBV2 MLP subgraph generator made atlas-size agnostic."""

    def __init__(self, node_features_num: int, edge_features_num: int, device: torch.device):
        super().__init__()
        self.device = device
        self.node_features_num = node_features_num
        self.edge_features_num = edge_features_num
        self.feature_size = 64
        self.linear = nn.Linear(self.node_features_num, self.feature_size).to(self.device)
        self.linear1 = nn.Linear(2 * self.feature_size, 8).to(self.device)
        self.linear2 = nn.Linear(8, 1).to(self.device)

    def _sample_graph(self, sampling_weights: torch.Tensor, temperature: float = 1.0, bias: float = 0.0) -> torch.Tensor:
        if self.training:
            bias = bias + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size(), device=self.device) + (1 - bias)
            gate_inputs = (torch.log(eps) - torch.log(1 - eps) + sampling_weights) / temperature
            return torch.sigmoid(gate_inputs)
        return torch.sigmoid(sampling_weights)

    def _edge_prob_mat(self, graph: Data) -> torch.Tensor:
        graph = graph.to(self.device)
        node_num = graph.x.shape[0]
        x = self.linear(graph.x).to(self.device)
        f1 = x.unsqueeze(1).repeat(1, node_num, 1).view(-1, self.feature_size)
        f2 = x.unsqueeze(0).repeat(node_num, 1, 1).view(-1, self.feature_size)
        f12self = torch.cat([f1, f2], dim=-1)
        f12self = torch.sigmoid(self.linear2(torch.sigmoid(self.linear1(f12self))))
        mask_sigmoid = f12self.reshape(node_num, node_num)
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edgemask = sym_mask[graph.edge_index[0], graph.edge_index[1]]
        return self._sample_graph(edgemask, temperature=0.5, bias=0.0)

    def forward(self, graph: Data) -> Tuple[Data, torch.Tensor]:
        subgraph = graph.to(self.device)
        edge_prob_matrix = self._edge_prob_mat(subgraph)
        # 修改备注：保持原 BrainIBV2 代码的子图生成接口，仅将硬编码 116 节点改为动态节点数；
        # 不改变原算法中把采样边权保存到 attr、并用方差作为 positive penalty 的核心逻辑。
        subgraph.attr = edge_prob_matrix
        pos_penalty = edge_prob_matrix.var()
        return subgraph, pos_penalty


class FederatedBrainIB(nn.Module):
    def __init__(self, num_nodes: int, device: torch.device):
        super().__init__()
        self.gnn = FedBrainIBGNN(num_nodes, device)
        self.subgraph = FedMLPSubgraph(num_nodes, 1, device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label_from_subject(subject_name: str) -> int:
    if subject_name.startswith("sub-control"):
        return 0
    if subject_name.startswith("sub-patient"):
        return 1
    raise ValueError(f"Cannot infer label from subject folder name: {subject_name}")


def subject_id_from_name(subject_name: str) -> str:
    return "".join(ch for ch in subject_name if ch.isdigit())


def load_correlation_matrix(mat_path: Path) -> np.ndarray:
    data = scio.loadmat(mat_path)
    matrix_keys = [key for key, value in data.items() if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2]
    if not matrix_keys:
        raise ValueError(f"No 2-D matrix found in {mat_path}")
    square_keys = [key for key in matrix_keys if data[key].shape[0] == data[key].shape[1]]
    key = square_keys[0] if square_keys else matrix_keys[0]
    matrix = np.asarray(data[key], dtype=np.float32)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Correlation matrix must be square in {mat_path}, got {matrix.shape}")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def matrix_to_graph(matrix: np.ndarray, label: int, site: str, subject: str, top_r: float) -> Data:
    node_num = matrix.shape[0]
    upper_i, upper_j = np.triu_indices(node_num, k=1)
    weights = matrix[upper_i, upper_j]
    edge_count = max(1, int(math.ceil(len(weights) * top_r)))
    selected = np.argpartition(np.abs(weights), -edge_count)[-edge_count:]
    src = upper_i[selected]
    dst = upper_j[selected]
    selected_weights = weights[selected]
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_attr = np.concatenate([selected_weights, selected_weights]).astype(np.float32)
    graph = Data(
        x=torch.tensor(matrix, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1),
        y=torch.tensor([label], dtype=torch.long),
    )
    graph.site = site
    graph.subject = subject
    return graph


def load_abide(root: Path, parcellation: str, top_r: float) -> Dict[str, List[Data]]:
    phenotype_path = root / "Phenotypic_V1_0b.csv"
    if not phenotype_path.exists():
        raise FileNotFoundError(f"ABIDE phenotype file not found: {phenotype_path}")
    sub_to_site = {}
    with phenotype_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sub_to_site[str(row["SUB_ID"]).strip()] = str(row["SITE_ID"]).strip()

    site_graphs = {site: [] for site in ABIDE_SITES}
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = parse_label_from_subject(subject_dir.name)
        subject_id = subject_id_from_name(subject_dir.name)
        site = sub_to_site.get(subject_id)
        if site is None:
            print(f"[WARN] Skip {subject_dir.name}: SUB_ID {subject_id} not found in phenotype CSV", file=sys.stderr)
            continue
        mat_path = subject_dir / f"{subject_dir.name}_{parcellation}_correlation_matrix.mat"
        if not mat_path.exists():
            print(f"[WARN] Skip {subject_dir.name}: missing {mat_path.name}", file=sys.stderr)
            continue
        graph = matrix_to_graph(load_correlation_matrix(mat_path), label, site, subject_dir.name, top_r)
        site_graphs.setdefault(site, []).append(graph)
    return {site: graphs for site, graphs in site_graphs.items() if graphs}


def load_adhd(root: Path, parcellation: str, top_r: float) -> Dict[str, List[Data]]:
    site_graphs = {}
    for site in ADHD_SITES:
        site_dir = root / site
        if not site_dir.exists():
            print(f"[WARN] ADHD site directory not found: {site_dir}", file=sys.stderr)
            continue
        graphs = []
        for subject_dir in sorted(path for path in site_dir.iterdir() if path.is_dir()):
            label = parse_label_from_subject(subject_dir.name)
            mat_path = subject_dir / f"{subject_dir.name}_{parcellation}_correlation_matrix.mat"
            if not mat_path.exists():
                print(f"[WARN] Skip {site}/{subject_dir.name}: missing {mat_path.name}", file=sys.stderr)
                continue
            graphs.append(matrix_to_graph(load_correlation_matrix(mat_path), label, site, subject_dir.name, top_r))
        if graphs:
            site_graphs[site] = graphs
    return site_graphs


def stratified_client_split(graphs: Sequence[Data], val_ratio: float, seed: int) -> Tuple[List[Data], List[Data]]:
    labels = [int(graph.y.item()) for graph in graphs]
    indices = np.arange(len(graphs))
    if len(graphs) < 2:
        return list(graphs), []
    unique, counts = np.unique(labels, return_counts=True)
    val_count = max(1, int(round(len(graphs) * val_ratio)))
    val_count = min(val_count, len(graphs) - 1)
    can_stratify = (
        len(unique) > 1
        and np.min(counts) >= 2
        and val_count >= len(unique)
        and (len(graphs) - val_count) >= len(unique)
    )
    stratify = labels if can_stratify else None
    train_idx, val_idx = train_test_split(indices, test_size=val_count, random_state=seed, shuffle=True, stratify=stratify)
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[0], -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(float(sigma), 1e-8)
    return torch.exp(-pairwise_distances(x) / sigma)


def renyi_entropy(x: torch.Tensor, sigma: float) -> torch.Tensor:
    alpha = 5
    k = calculate_gram_mat(x, sigma)
    k = k / torch.clamp(torch.trace(k), min=1e-8)
    eigv = torch.abs(torch.linalg.eigvalsh(k))
    eig_pow = eigv ** alpha
    return (1 / (1 - alpha)) * torch.log2(torch.clamp(torch.sum(eig_pow), min=1e-8))


def joint_entropy(x: torch.Tensor, y: torch.Tensor, s_x: float, s_y: float) -> torch.Tensor:
    alpha = 5
    x_gram = calculate_gram_mat(x, s_x)
    y_gram = calculate_gram_mat(y, s_y)
    k = torch.mul(x_gram, y_gram)
    k = k / torch.clamp(torch.trace(k), min=1e-8)
    eigv = torch.abs(torch.linalg.eigvalsh(k))
    eig_pow = eigv ** alpha
    return (1 / (1 - alpha)) * torch.log2(torch.clamp(torch.sum(eig_pow), min=1e-8))


def calculate_mi(x: torch.Tensor, y: torch.Tensor, s_x: float, s_y: float) -> torch.Tensor:
    return renyi_entropy(x, s_x) + renyi_entropy(y, s_y) - joint_entropy(x, y, s_x, s_y)


def sigma_from_embeddings(embeddings: torch.Tensor) -> float:
    if embeddings.shape[0] <= 1:
        return 1.0
    z_numpy = embeddings.detach().cpu().numpy()
    distances = squareform(pdist(z_numpy, "euclidean"))
    distances = distances[~np.eye(distances.shape[0], dtype=bool)].reshape(distances.shape[0], -1)
    k = min(10, distances.shape[1])
    sigma = float(np.mean(np.sort(distances, axis=1)[:, :k]))
    return max(sigma ** 2, 1e-8)


def local_train(args: argparse.Namespace, model: FederatedBrainIB, train_dataset: Sequence[Data], device: torch.device) -> float:
    if not train_dataset:
        return 0.0
    model.train()
    optimizer = torch.optim.Adam([
        {"params": model.gnn.parameters(), "lr": args.model_learning_rate},
        {"params": model.subgraph.parameters(), "lr": args.SGmodel_learning_rate},
    ])
    criterion = nn.CrossEntropyLoss()
    loss_accum = 0.0
    step_count = 0
    for _ in range(args.local_epochs):
        shuffled = list(train_dataset)
        random.shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            graphs = shuffled[start:start + args.batch_size]
            batch_graph = next(iter(DataLoader(graphs, batch_size=len(graphs))))
            embeddings, _ = model.gnn(batch_graph)
            subgraphs = copy.deepcopy(graphs)
            for graph in subgraphs:
                model.subgraph(graph)
            batch_subgraph = next(iter(DataLoader(subgraphs, batch_size=len(subgraphs))))
            positive, subgraph_output = model.gnn(batch_subgraph)
            mi_loss = calculate_mi(embeddings, positive, sigma_from_embeddings(embeddings), sigma_from_embeddings(positive)) / len(graphs)
            labels = batch_graph.y.view(-1).to(device)
            classify_loss = criterion(subgraph_output, labels)
            loss = classify_loss + mi_loss * args.mi_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_accum += float(loss.detach().cpu())
            step_count += 1
    return loss_accum / max(step_count, 1)


def fedavg_state(states: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    averaged = OrderedDict()
    for key in states[0].keys():
        averaged[key] = sum(state[key].detach().cpu() * weight for state, weight in zip(states, weights))
    return averaged


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    if not y_true:
        return {name: float("nan") for name in METRIC_NAMES}
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) < 2:
        metrics["auc"] = float("nan")
    else:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    return metrics


def evaluate(model: FederatedBrainIB, dataset: Sequence[Data], batch_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for batch in DataLoader(list(dataset), batch_size=batch_size):
            _, logits = model.gnn(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            labels = batch.y.view(-1).to(device)
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().astype(int).tolist())
            y_score.extend(probs[:, 1].cpu().numpy().tolist())
    return compute_metrics(y_true, y_pred, y_score)


def macro_client_auc(model: FederatedBrainIB, client_val: Dict[str, List[Data]], batch_size: int, device: torch.device) -> float:
    aucs = []
    for val_dataset in client_val.values():
        if val_dataset:
            auc = evaluate(model, val_dataset, batch_size, device)["auc"]
            if not math.isnan(auc):
                aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def merge_client_data(client_data: Dict[str, List[Data]]) -> List[Data]:
    merged = []
    for graphs in client_data.values():
        merged.extend(graphs)
    return merged


def run_holdout(args: argparse.Namespace, site_graphs: Dict[str, List[Data]], test_site: str, num_nodes: int, device: torch.device) -> Dict[str, float]:
    train_sites = [site for site in sorted(site_graphs) if site != test_site]
    client_train, client_val = {}, {}
    for site in train_sites:
        train_split, val_split = stratified_client_split(site_graphs[site], args.val_ratio, args.seed)
        client_train[site] = train_split
        client_val[site] = val_split

    global_model = FederatedBrainIB(num_nodes, device).to(device)
    best_state = copy.deepcopy(global_model.state_dict())
    best_round = 0
    best_val_auc = -float("inf")
    rounds_without_improve = 0

    for round_idx in range(1, args.rounds + 1):
        local_states, local_weights, local_losses = [], [], []
        total_samples = sum(len(graphs) for graphs in client_train.values())
        for site, train_dataset in client_train.items():
            if not train_dataset:
                continue
            local_model = FederatedBrainIB(num_nodes, device).to(device)
            local_model.load_state_dict(global_model.state_dict())
            loss = local_train(args, local_model, train_dataset, device)
            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(len(train_dataset) / total_samples)
            local_losses.append(loss)
        if local_states:
            global_model.load_state_dict(fedavg_state(local_states, local_weights))

        train_metrics = evaluate(global_model, merge_client_data(client_train), args.batch_size, device)
        val_metrics = evaluate(global_model, merge_client_data(client_val), args.batch_size, device)
        test_metrics = evaluate(global_model, site_graphs[test_site], args.batch_size, device)
        val_macro_auc = macro_client_auc(global_model, client_val, args.batch_size, device)
        print(
            f"[{test_site}] round {round_idx:03d} loss={np.mean(local_losses) if local_losses else 0.0:.5f} "
            f"train_acc={train_metrics['accuracy']:.5f} val_auc={val_metrics['auc']:.5f} "
            f"macro_val_auc={val_macro_auc:.5f} test_auc={test_metrics['auc']:.5f}"
        )

        comparable_auc = val_macro_auc if not math.isnan(val_macro_auc) else -float("inf")
        if round_idx >= args.min_rounds and comparable_auc > best_val_auc:
            best_val_auc = comparable_auc
            best_round = round_idx
            best_state = copy.deepcopy(global_model.state_dict())
            rounds_without_improve = 0
        elif round_idx >= args.min_rounds:
            rounds_without_improve += 1
            if args.patience > 0 and rounds_without_improve >= args.patience:
                print(f"[{test_site}] early stopping at round {round_idx}; best_round={best_round}, best_macro_val_auc={best_val_auc:.5f}")
                break

    global_model.load_state_dict(best_state)
    final_train = evaluate(global_model, merge_client_data(client_train), args.batch_size, device)
    final_val = evaluate(global_model, merge_client_data(client_val), args.batch_size, device)
    final_test = evaluate(global_model, site_graphs[test_site], args.batch_size, device)
    result = {"test_site": test_site, "best_round": best_round, "best_val_macro_auc": best_val_auc}
    for prefix, metrics in (("train", final_train), ("val", final_val), ("test", final_test)):
        for name, value in metrics.items():
            result[f"{prefix}_{name}"] = value
    return result


def format_metric(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def write_results_csv(output_dir: Path, results: List[Dict[str, float]]) -> None:
    metric_columns = [f"{prefix}_{metric}" for prefix in ("train", "val", "test") for metric in METRIC_NAMES]
    fieldnames = ["test_site", "best_round", "best_val_macro_auc"] + metric_columns
    rows = list(results)
    for stat_name, reducer in (("mean", np.nanmean), ("std", np.nanstd)):
        row = {"test_site": stat_name, "best_round": "", "best_val_macro_auc": reducer([r["best_val_macro_auc"] for r in results])}
        for column in metric_columns:
            row[column] = reducer([r[column] for r in results])
        rows.append(row)

    output_path = output_dir / "site_holdout_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_metric(row.get(key, "")) for key in fieldnames})
    print(f"Saved results to {output_path}")


def run_experiment(args: argparse.Namespace, dataset_name: str, parcellation: str, device: torch.device) -> None:
    top_r = args.top_r / 100.0 if args.top_r > 1 else args.top_r
    if dataset_name == "abide":
        site_graphs = load_abide(Path(args.abide_root), parcellation, top_r)
        test_sites = ABIDE_TEST_SITES
    elif dataset_name == "adhd":
        site_graphs = load_adhd(Path(args.adhd_root), parcellation, top_r)
        test_sites = ADHD_TEST_SITES
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_nodes = PARCELLATIONS[parcellation]
    missing = [site for site in test_sites if site not in site_graphs]
    if missing:
        raise ValueError(f"Missing required test sites for {dataset_name}/{parcellation}: {missing}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_root) / f"{dataset_name}_{parcellation}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {dataset_name}/{parcellation}; output={output_dir}")

    results = []
    for test_site in test_sites:
        results.append(run_holdout(args, site_graphs, test_site, num_nodes, device))
    write_results_csv(output_dir, results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated BrainIBV2 for ABIDE and ADHD site-held-out experiments")
    parser.add_argument("--abide_root", type=str, default=r"D:\Pycharm\Brain_Generalization_Projects\BrainGeneralization-main\data\preprocessed_data\abide")
    parser.add_argument("--adhd_root", type=str, default=r"D:\Pycharm\Brain_Generalization_Projects\BrainGeneralization-main\data\preprocessed_data\adhd_balanced_labels")
    parser.add_argument("--output_root", type=str, default="fed_results")
    parser.add_argument("--datasets", nargs="+", default=["abide", "adhd"], choices=["abide", "adhd"])
    parser.add_argument("--parcellations", nargs="+", default=["harvard48", "schaefer100", "AAL116"], choices=list(PARCELLATIONS.keys()))
    parser.add_argument("--top_r", type=float, default=0.20, help="Top-r absolute-correlation edge ratio. Use 0.20 or 20 for 20%%.")
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--min_rounds", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience after min_rounds; <=0 disables early stopping.")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mi_weight", type=float, default=0.001)
    parser.add_argument("--model_learning_rate", type=float, default=0.0005)
    parser.add_argument("--SGmodel_learning_rate", type=float, default=0.001)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset_name in args.datasets:
        for parcellation in args.parcellations:
            run_experiment(args, dataset_name, parcellation, device)


if __name__ == "__main__":
    main()
