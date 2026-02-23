from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import dns.resolver
import tldextract
import torch
from torch_geometric.data import HeteroData


@dataclass
class GraphBuilderConfig:
    max_ips: int = 8
    max_nameservers: int = 8


class GraphBuilder:
    def __init__(self, config: Optional[GraphBuilderConfig] = None) -> None:
        self.config = config or GraphBuilderConfig()

    def build_subgraph(self, url: str) -> HeteroData:
        parsed = urlparse(url.strip())
        host = parsed.hostname or ""
        domain = self._extract_domain(host)
        ips = self._resolve_ips(domain) if domain else set()
        nameservers = self._resolve_nameservers(domain) if domain else set()

        url_nodes: List[str] = [url]
        domain_nodes: List[str] = [domain] if domain else []
        ip_nodes: List[str] = list(ips)[: self.config.max_ips]
        nameserver_nodes: List[str] = list(nameservers)[: self.config.max_nameservers]

        data = HeteroData()

        data["url"].num_nodes = len(url_nodes)
        data["url"].url = url_nodes

        data["domain"].num_nodes = len(domain_nodes)
        data["domain"].name = domain_nodes

        data["ip_address"].num_nodes = len(ip_nodes)
        data["ip_address"].address = ip_nodes

        data["nameserver"].num_nodes = len(nameserver_nodes)
        data["nameserver"].hostname = nameserver_nodes

        edge_index_dict: Dict[str, torch.Tensor] = {}

        if url_nodes and domain_nodes:
            url_to_domain = torch.tensor([[0], [0]], dtype=torch.long)
            edge_index_dict[("url", "belongs_to", "domain")] = url_to_domain

        if domain_nodes and ip_nodes:
            domain_indices = []
            ip_indices = []
            for idx, _ in enumerate(ip_nodes):
                domain_indices.append(0)
                ip_indices.append(idx)
            edge_index = torch.tensor(
                [domain_indices, ip_indices],
                dtype=torch.long,
            )
            edge_index_dict[("domain", "resolves_to", "ip_address")] = edge_index

        if domain_nodes and nameserver_nodes:
            domain_indices = []
            ns_indices = []
            for idx, _ in enumerate(nameserver_nodes):
                domain_indices.append(0)
                ns_indices.append(idx)
            edge_index = torch.tensor(
                [domain_indices, ns_indices],
                dtype=torch.long,
            )
            edge_index_dict[("domain", "managed_by", "nameserver")] = edge_index

        for edge_type, edge_index in edge_index_dict.items():
            data[edge_type].edge_index = edge_index

        return data

    def _extract_domain(self, host: str) -> str:
        if not host:
            return ""
        extracted = tldextract.extract(host)
        if not extracted.domain or not extracted.suffix:
            return host
        return f"{extracted.domain}.{extracted.suffix}"

    def _resolve_ips(self, domain: str) -> Set[str]:
        results: Set[str] = set()
        if not domain:
            return results
        try:
            info = socket.getaddrinfo(domain, None)
        except OSError:
            return results
        for entry in info:
            sockaddr = entry[4]
            if sockaddr and len(sockaddr) >= 1:
                ip = sockaddr[0]
                if ip:
                    results.add(ip)
        return results

    def _resolve_nameservers(self, domain: str) -> Set[str]:
        results: Set[str] = set()
        if not domain:
            return results
        try:
            answers = dns.resolver.resolve(domain, "NS")
        except Exception:
            return results
        for record in answers:
            value = str(record.target).rstrip(".")
            if value:
                results.add(value)
        return results

