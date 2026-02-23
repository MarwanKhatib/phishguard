from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import redis
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


class GraphEmbeddingCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "phishshield:gnn",
    ) -> None:
        self.prefix = prefix
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)

    def _key(self, node_type: str, node_id: str) -> str:
        return f"{self.prefix}:{node_type}:{node_id}"

    def get_embedding(self, node_type: str, node_id: str) -> Optional[torch.Tensor]:
        key = self._key(node_type, node_id)
        value = self.client.get(key)
        if value is None:
            return None
        try:
            array = np.array(json.loads(value.decode("utf-8")), dtype=np.float32)
        except Exception:
            return None
        tensor = torch.from_numpy(array)
        return tensor

    def set_embedding(
        self,
        node_type: str,
        node_id: str,
        embedding: torch.Tensor,
    ) -> None:
        key = self._key(node_type, node_id)
        array = embedding.detach().cpu().numpy().astype(np.float32)
        payload = json.dumps(array.tolist()).encode("utf-8")
        self.client.set(key, payload)


class URLCharEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        num_layers: int = 2,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        vocab = self._build_vocab()
        self.stoi = vocab
        self.pad_index = 0
        self.embedding = nn.Embedding(len(self.stoi), d_model)
        self.positional = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_vocab(self) -> Dict[str, int]:
        chars = [chr(i) for i in range(32, 127)]
        stoi: Dict[str, int] = {}
        index = 0
        stoi["<pad>"] = index
        index += 1
        for ch in chars:
            if ch not in stoi:
                stoi[ch] = index
                index += 1
        return stoi

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        batch_size = len(texts)
        ids = torch.full(
            (batch_size, self.max_length),
            self.pad_index,
            dtype=torch.long,
        )
        for i, text in enumerate(texts):
            truncated = text[: self.max_length]
            for j, ch in enumerate(truncated):
                ids[i, j] = self.stoi.get(ch, self.pad_index)
        return ids

    def forward(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.embedding.embedding_dim)
        token_ids = self._encode_texts(texts)
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        positions = torch.arange(
            0,
            token_ids.size(1),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand_as(token_ids)
        x = self.embedding(token_ids) + self.positional(positions)
        padding_mask = token_ids.eq(self.pad_index)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        mask = (~padding_mask).float()
        mask_sum = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        return pooled


class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        homogeneous, node_types, _ = data.to_homogeneous(return_types=True)
        device = homogeneous.edge_index.device
        num_nodes = homogeneous.num_nodes
        edge_index = homogeneous.edge_index
        if edge_index.numel() == 0 or num_nodes == 0:
            return {node_type: torch.zeros(0, self.dim) for node_type in set(node_types)}
        row = edge_index[0]
        col = edge_index[1]
        indices = torch.cat(
            [
                torch.stack([row, col], dim=0),
                torch.stack([col, row], dim=0),
            ],
            dim=1,
        )
        values = torch.ones(indices.size(1), device=device)
        adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=(num_nodes, num_nodes),
        ).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_left = deg_inv_sqrt.view(-1, 1)
        d_right = deg_inv_sqrt.view(1, -1)
        dense_adj = adj.to_dense()
        laplacian = torch.eye(num_nodes, device=device) - d_left * dense_adj * d_right
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        except RuntimeError:
            eigenvectors = torch.zeros(num_nodes, self.dim, device=device)
            return self._split_by_type(eigenvectors, node_types)
        k = min(self.dim, eigenvectors.size(1))
        embeddings = eigenvectors[:, :k]
        if k < self.dim:
            padding = torch.zeros(num_nodes, self.dim - k, device=device)
            embeddings = torch.cat([embeddings, padding], dim=1)
        return self._split_by_type(embeddings, node_types)

    def _split_by_type(
        self,
        embeddings: torch.Tensor,
        node_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        result: Dict[str, List[torch.Tensor]] = {}
        for idx, node_type in enumerate(node_types):
            result.setdefault(node_type, []).append(embeddings[idx])
        output: Dict[str, torch.Tensor] = {}
        for node_type, items in result.items():
            if items:
                output[node_type] = torch.stack(items, dim=0)
            else:
                output[node_type] = torch.zeros(0, embeddings.size(1))
        return output


@dataclass
class HeteroGTConfig:
    hidden_dim: int = 128
    lpe_dim: int = 16
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    use_embedding_cache: bool = True
    redis_url: str = "redis://localhost:6379/0"


class HeteroGTModel(nn.Module):
    def __init__(self, config: Optional[HeteroGTConfig] = None) -> None:
        super().__init__()
        self.config = config or HeteroGTConfig()
        url_dim = self.config.hidden_dim - self.config.lpe_dim
        self.url_encoder = URLCharEncoder(
            d_model=url_dim,
            n_heads=self.config.num_heads,
            num_layers=2,
            max_length=256,
        )
        self.lpe = LaplacianPositionalEncoding(dim=self.config.lpe_dim)
        node_types = ["url", "domain", "ip_address", "nameserver"]
        self.node_types = node_types
        self.type_projections = nn.ModuleDict()
        for node_type in node_types:
            if node_type == "url":
                in_dim = self.config.lpe_dim + url_dim
            else:
                in_dim = self.config.lpe_dim
            self.type_projections[node_type] = nn.Linear(in_dim, self.config.hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(self.config.num_layers):
            conv = HGTConv(
                in_channels=self.config.hidden_dim,
                out_channels=self.config.hidden_dim,
                metadata=(node_types, self._edge_types()),
                heads=self.config.num_heads,
            )
            self.convs.append(conv)
        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.cache: Optional[GraphEmbeddingCache] = None
        if self.config.use_embedding_cache:
            self.cache = GraphEmbeddingCache(redis_url=self.config.redis_url)

    def _edge_types(self) -> List[Tuple[str, str, str]]:
        return [
            ("url", "belongs_to", "domain"),
            ("domain", "resolves_to", "ip_address"),
            ("domain", "managed_by", "nameserver"),
        ]

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        lpe_dict = self.lpe(data)
        for key in lpe_dict:
            lpe_dict[key] = lpe_dict[key].to(device)
        url_texts: List[str] = list(getattr(data["url"], "url", []))
        url_embeddings = self.url_encoder(url_texts).to(device)
        x_dict: Dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            num_nodes = data[node_type].num_nodes
            lpe = lpe_dict.get(node_type)
            if lpe is None or lpe.size(0) != num_nodes:
                lpe = torch.zeros(num_nodes, self.config.lpe_dim, device=device)
            if node_type == "url":
                if url_embeddings.size(0) != num_nodes:
                    url_embeddings = torch.zeros(
                        num_nodes,
                        self.config.hidden_dim - self.config.lpe_dim,
                        device=device,
                    )
                features = torch.cat([lpe, url_embeddings], dim=-1)
            else:
                features = lpe
            projected = self.type_projections[node_type](features)
            x_dict[node_type] = projected
        if self.cache is not None:
            self._inject_cached_embeddings(data, x_dict)
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            for key in x_dict:
                x_dict[key] = self.dropout(x_dict[key])
        url_repr = x_dict["url"]
        logits = self.mlp(url_repr).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        heatmap = self._compute_heatmap(x_dict)
        if self.cache is not None:
            self._update_cache(data, x_dict)
        return probabilities, heatmap

    def _compute_heatmap(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        heatmap: Dict[str, torch.Tensor] = {}
        for node_type in ["domain", "ip_address", "nameserver"]:
            embeddings = x_dict.get(node_type)
            if embeddings is None or embeddings.size(0) == 0:
                heatmap[node_type] = torch.zeros(0)
                continue
            scores = embeddings.norm(p=2, dim=-1)
            max_score = scores.max()
            if max_score > 0:
                scores = scores / max_score
            heatmap[node_type] = scores
        return heatmap

    def _inject_cached_embeddings(
        self,
        data: HeteroData,
        x_dict: Dict[str, torch.Tensor],
    ) -> None:
        if self.cache is None:
            return
        device = next(self.parameters()).device
        for node_type in ["domain", "ip_address", "nameserver"]:
            embeddings = x_dict.get(node_type)
            if embeddings is None or embeddings.size(0) == 0:
                continue
            names: Iterable[str]
            if node_type == "domain":
                names = getattr(data[node_type], "name", [])
            elif node_type == "ip_address":
                names = getattr(data[node_type], "address", [])
            else:
                names = getattr(data[node_type], "hostname", [])
            names_list = list(names)
            for idx, node_id in enumerate(names_list):
                cached = self.cache.get_embedding(node_type, node_id)
                if cached is not None and cached.numel() == embeddings.size(1):
                    x_dict[node_type][idx] = cached.to(device)

    def _update_cache(self, data: HeteroData, x_dict: Dict[str, torch.Tensor]) -> None:
        if self.cache is None:
            return
        for node_type in ["domain", "ip_address", "nameserver"]:
            embeddings = x_dict.get(node_type)
            if embeddings is None or embeddings.size(0) == 0:
                continue
            names: Iterable[str]
            if node_type == "domain":
                names = getattr(data[node_type], "name", [])
            elif node_type == "ip_address":
                names = getattr(data[node_type], "address", [])
            else:
                names = getattr(data[node_type], "hostname", [])
            names_list = list(names)
            for idx, node_id in enumerate(names_list):
                self.cache.set_embedding(node_type, node_id, embeddings[idx])

