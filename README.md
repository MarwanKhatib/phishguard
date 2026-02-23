# PhishShield 2026 – Graph Edition (GNN + Graph Transformer)

PhishShield is a production‑grade, enterprise‑ready phishing detection engine
designed for Security Operations Centers (SOCs). The **Graph Edition** analyzes
URLs not only by their text, but also by their relationships to global internet
infrastructure (domains, IPs, name servers) using a heterogeneous Graph
Transformer.

Core technologies:

- Python 3.11+
- PyTorch + PyTorch Geometric (heterogeneous GNN / Graph Transformer)
- Redis (graph embedding cache)
- PostgreSQL (graph entities and relationships, not yet wired in this skeleton)
- Django + Django REST Framework + Celery (for asynchronous graph construction
  and API integration; to be added on top)

This repository currently provides:

- URL cleaning pipeline from multiple raw datasets (`datasets/build_training_dataset.py`)
- Core feature extraction engine (`engine.extractor.FeatureExtractor`) for
  lexical/infrastructure analysis (optional)
- Heterogeneous graph builder (`engine.graph.GraphBuilder`)
- Heterogeneous Graph Transformer model (`engine.graph.HeteroGTModel`) with:
  - Character‑level Transformer encoder for URL nodes
  - Laplacian positional encodings (LPE) for structural roles
  - Graph attention over URL–Domain–IP–NameServer relationships (HGTConv)
  - Redis‑backed graph embedding cache for domains, IPs, and name servers
- Docker Compose stack for Django + Celery + Redis + PostgreSQL

The Django API, Celery tasks, Playwright integration, and Postgres persistence
can be layered on top of this engine following the blueprint described below.

---

## 1. Implemented Features

### 1.1 Lexical URL Features

All lexical features are computed directly from the full URL string:

- `url_length`: Total length of the URL.
- `url_entropy`: Shannon entropy over all characters in the URL.
- `dot_density`: Ratio of `.` characters to total length.
- `hyphen_density`: Ratio of `-` characters to total length.
- `ratio_dollar`: Ratio of `$` characters.
- `ratio_at`: Ratio of `@` characters.
- `ratio_underscore`: Ratio of `_` characters.
- `ratio_question`: Ratio of `?` characters.
- `ratio_equal`: Ratio of `=` characters.

These features capture obfuscation patterns, noisy tracking parameters, and suspicious
tokenization commonly used in phishing URLs.

### 1.2 Infrastructure Forensics

The engine performs several low‑level checks on the host and domain:

- `uses_standard_ip`:
  - Detects whether the hostname is a direct IPv4 or IPv6 address
    (`ipaddress.ip_address`).

- `uses_encoded_ip`:
  - Detects hexadecimal, octal, or decimal encoded IP formats using regex and numeric
    conversion, then validates with `ipaddress.ip_address`.

- `asn_trust_score`:
  - If the host is an IP, uses `ipwhois.IPWhois.lookup_rdap` to obtain ASN metadata.
  - Produces a heuristic trust score in `[0, 1]` based on ASN description keywords,
    penalizing “hosting / VPN / proxy / bulletproof” style networks and slightly
    boosting well‑known cloud providers.

- `domain_age_days`:
  - Uses `python-whois` to obtain `creation_date` for the registrable domain.
  - Computes age in days as `now - creation_date`.

These features help distinguish long‑standing legitimate infrastructure from newly
registered or low‑reputation hosts.

### 1.3 Advanced URL Parameter Analysis

The query string is parsed into key/value pairs and analyzed for:

- `has_email_in_params`:
  - Detects credential pre‑filling by searching for email patterns in parameter values.

- `has_base64_param_value`:
  - Heuristically identifies Base64‑encoded values:
    - Length check and multiple‑of‑4 constraint.
    - Character set validation.
    - Successful `base64.b64decode(..., validate=True)`.
    - At least ~70% printable characters in decoded payload.

- `has_open_redirect_param`:
  - Examines parameter names for open redirect keywords:
    - `next`, `url`, `redirect`, `redir`, `return`, `dest`.
  - Flags as suspicious when the corresponding value starts with `http://` or
    `https://`.

These features directly map to high‑impact phishing behaviors such as credential
harvesting and open redirect abuse.

### 1.4 Anti‑Evasion & Behavioral Signals

The engine computes several anti‑evasion indicators:

- Redirect chain metrics:
  - Uses `requests.Session().get(..., allow_redirects=True)` (with timeouts and a cap
    on maximum redirects) to derive:
    - `redirect_chain_length`: Number of redirects observed.
    - `final_url`: Final resolved URL (stored in metadata).
    - `uses_known_shortener`: Boolean flag if the final host is a known shortener
      (e.g., `bit.ly`, `t.co`, `tinyurl.com`).

- Brand‑jacking detection (`brand_jacking_flag`):
  - Uses `tldextract` to split `subdomain.domain.suffix`.
  - Searches for high‑value brand keywords in:
    - Subdomain (e.g., `paypal-secure.example.com`).
    - Domain where the domain is not the official brand (e.g., `paypal-login-bank.com`).

- Password field on low‑reputation sites
  (`password_field_on_low_reputation`):
  - Accepts an optional rendered HTML string (e.g., from Playwright).
  - Searches for `<input type="password">` via regex.
  - Flags as suspicious when the domain age is low (young/unknown domain).

All of these are aggregated into a fixed‑order numeric feature vector alongside a
metadata dictionary used by the XAI layer.

### 1.5 Feature Vector Layout

`engine.extractor.FeatureExtractor` outputs:

- `FeatureExtractionResult.vector`: `numpy.ndarray` of `float`, with features in the
  following order:

  1. `url_length`
  2. `url_entropy`
  3. `dot_density`
  4. `hyphen_density`
  5. `ratio_dollar`
  6. `ratio_at`
  7. `ratio_underscore`
  8. `ratio_question`
  9. `ratio_equal`
  10. `uses_standard_ip`
  11. `uses_encoded_ip`
  12. `asn_trust_score`
  13. `domain_age_days`
  14. `has_email_in_params`
  15. `has_base64_param_value`
  16. `has_open_redirect_param`
  17. `redirect_chain_length`
  18. `uses_known_shortener`
  19. `brand_jacking_flag`
  20. `password_field_on_low_reputation`

- `FeatureExtractionResult.feature_names`: list of the feature names above.
- `FeatureExtractionResult.metadata`: extended context (URL, host, domain, final URL,
  domain age, ASN trust score, and individual boolean flags).

---

## 2. Graph Model & GNN Layer

### 2.1 Heterogeneous Graph Schema

PhishShield models the URL and its infrastructure as a small heterogeneous graph
using PyTorch Geometric’s `HeteroData`:

- Node types:
  - `url`
  - `domain`
  - `ip_address`
  - `nameserver`

- Edge types:
  - `("url", "belongs_to", "domain")`
  - `("domain", "resolves_to", "ip_address")`
  - `("domain", "managed_by", "nameserver")`

The **GraphBuilder** is responsible for constructing a 1‑hop subgraph for each
submitted URL.

### 2.2 GraphBuilder (URL → 1‑Hop Subgraph)

`engine.graph.GraphBuilder` builds a small heterogeneous graph from one URL by:

- Parsing the URL and extracting the registrable domain via `tldextract`.
- Resolving:
  - A set of IP addresses for the domain (via `socket.getaddrinfo`).
  - A set of name servers for the domain (via `dns.resolver`).
- Creating node sets:
  - `url.url`: list of original URLs (typically one per subgraph).
  - `domain.name`: list of domain names.
  - `ip_address.address`: list of IP addresses.
  - `nameserver.hostname`: list of NS hostnames.
- Wiring edges to match the schema above.

Example:

```python
from engine.graph import GraphBuilder

builder = GraphBuilder()
data = builder.build_subgraph("https://facebooken.login.piotrpiotr.pl/")

print(data)
print(data["url"].url)
print(data["domain"].name)
print(data["ip_address"].address)
print(data["nameserver"].hostname)
```

### 2.3 URLCharEncoder – Character‑Level Transformer for URL Nodes

`engine.graph.URLCharEncoder` (used internally by `HeteroGTModel`) encodes URL
strings into dense vectors using a mini Transformer:

- Builds a character vocabulary over printable ASCII plus `<pad>`.
- Truncates/pads each URL to a fixed max length (e.g., 256 characters).
- Applies:
  - Character embeddings.
  - Learned positional embeddings.
  - A stack of `nn.TransformerEncoderLayer`s (self‑attention).
- Mean‑pools the non‑padded tokens to obtain one embedding per URL.

This acts as a learned lexical feature extractor for the `url` nodes, replacing
hand‑engineered lexical features when using the pure GNN/GT pipeline.

### 2.4 Laplacian Positional Encodings (LPE)

`engine.graph.LaplacianPositionalEncoding` computes structural encodings for all
nodes:

- Converts the heterogeneous graph into a homogeneous graph using
  `HeteroData.to_homogeneous(return_types=True)`.
- Builds the symmetric normalized Laplacian:
  - `L = I - D^{-1/2} A D^{-1/2}`.
- Computes eigenvectors of `L` and takes the first `k` components as positional
  embeddings (Laplacian Eigenmaps).
- Splits the resulting embeddings back into per‑node‑type tensors using the
  `node_types` mapping from `to_homogeneous`.

These LPEs (dimension `lpe_dim`) preserve each node’s structural role in the
overall graph.

### 2.5 HeteroGTModel – Graph Transformer with HGTConv

`engine.graph.HeteroGTModel` combines:

- URL lexical embeddings from `URLCharEncoder`.
- Laplacian positional encodings for all node types.
- Type‑specific linear projections to a common hidden space.
- Multiple layers of `torch_geometric.nn.HGTConv` for message passing.
- A final MLP classifier over `url` node embeddings.

Configuration:

```python
from engine.graph import HeteroGTConfig, HeteroGTModel

config = HeteroGTConfig(
    hidden_dim=128,
    lpe_dim=16,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
    use_embedding_cache=True,
    redis_url="redis://localhost:6379/0",
)
model = HeteroGTModel(config=config)
```

Forward API:

```python
from engine.graph import GraphBuilder, HeteroGTModel, HeteroGTConfig
import torch

builder = GraphBuilder()
data = builder.build_subgraph("https://facebooken.login.piotrpiotr.pl/")

config = HeteroGTConfig()
model = HeteroGTModel(config=config)

model.eval()
with torch.no_grad():
    probabilities, heatmap = model(data)

print("Phish probability:", float(probabilities[0]))
```

Outputs:

- `probabilities`: tensor of phishing probabilities (0.0–1.0) for each URL node.
- `heatmap`: dict mapping node type to per‑node scores in `[0, 1]`:
  - `heatmap["domain"]` aligned with `data["domain"].name`
  - `heatmap["ip_address"]` aligned with `data["ip_address"].address`
  - `heatmap["nameserver"]` aligned with `data["nameserver"].hostname`

These heatmap scores act as a structural explanation: they highlight which
domains/IPs/name servers contributed most to the final score.

### 2.6 Graph Embedding Cache (Redis)

`engine.graph.GraphEmbeddingCache` provides a Redis‑based cache for infrastructure
node embeddings:

- Keys:
  - `phishshield:gnn:domain:{domain}`
  - `phishshield:gnn:ip_address:{ip}`
  - `phishshield:gnn:nameserver:{hostname}`
- Values:
  - JSON‑serialized float vectors (PyTorch tensors converted via NumPy).

`HeteroGTModel` uses this cache if `use_embedding_cache=True`:

- Before message passing:
  - For `domain`, `ip_address`, `nameserver` nodes, attempts to load cached
    embeddings and inject them into the feature tensors.
- After message passing:
  - Writes updated embeddings back to Redis, so future graphs can reuse the
    learned representation of known infrastructure nodes.

---

## 3. System Blueprint & Architecture

### 3.1 Core Components

- **Backend**: Django 5 + Django REST Framework
  - Exposes REST endpoints for URL scanning, history lookup, and admin operations.

- **AI Engine**:
  - `engine.graph.GraphBuilder`
  - `engine.graph.HeteroGTModel`
  - Optional: `engine.extractor.FeatureExtractor` for auxiliary lexical features.

- **Task Management**: Celery + Redis
  - Asynchronous “Deep Scans” that enrich initial decisions with WHOIS, DNS, and
    headless browser rendering (Playwright).

- **Storage**:
  - PostgreSQL (scan history, audit logs, retraining datasets).
  - Redis (cache‑aside: MD5 hash of URL → risk score + decision).

- **Infrastructure**:
  - Docker Compose with services: `web`, `worker`, `beat`, `redis`, `db`.

### 3.2 Enterprise Logic Flow

High‑level flow for a SOC request to “scan URL X”:

1. **MD5 Hash & Cache‑First Lookup**
   - Compute MD5 of the URL string.
   - Check Redis cache:
     - If hit: return cached risk score and decision immediately.
   - If miss: check PostgreSQL scan history for a recent record:
     - If recent: load last decision and return (optionally refreshing TTL in Redis).

2. **AI Engine Evaluation**
   - If still a miss:
     - Use `GraphBuilder` to construct a 1‑hop heterogeneous subgraph around the
       URL (URL–Domain–IP–NameServer).
     - Pass the subgraph to `HeteroGTModel` to obtain:
       - Phishing probability for the URL node.
       - Heatmap scores for domain/IP/NS nodes.
   - Persist this initial graph‑based decision to PostgreSQL and cache the result
     in Redis keyed by the MD5 hash.

3. **Deep Scan (Celery Task)**
   - Enqueue a Celery job with the URL and a reference to the initial scan record.
   - Deep scan may:
     - Perform additional WHOIS/DNS/ASN checks (or re‑confirm existing signals).
     - Use Playwright to render the page in a headless browser and collect final DOM.
     - Re‑run `FeatureExtractor` with real HTML to refine behavioral features
       (e.g., password fields, dynamic redirects).
     - Update the scan record in PostgreSQL with enriched features and DOM indicators.

4. **Audit Trail & Retraining**
   - Every scan attempt (initial and deep) is logged in PostgreSQL:
     - URL, MD5 hash, timestamps.
     - Graph snapshot metadata (domain/IP/NS sets, edge counts).
     - Model probability, heatmap scores, and final decision.
     - Enrichment metadata (WHOIS, ASN, rendering flags).
   - Offline jobs can export these records into datasets suitable for GNN training
     (URL, graph structure, and labels).

---

## 4. Running the Stack with Docker Compose

Prerequisites:

- Docker and Docker Compose installed.

Steps:

1. Build and start all services:

   ```bash
   docker-compose up --build
   ```

2. The Django API (once created) will be exposed on:

   - `http://localhost:8000/`

3. Typical endpoints you would implement:

   - `POST /api/scan/` – Submit a URL for scanning.
   - `GET /api/scan/{id}/` – Retrieve a scan result by ID.
   - `GET /api/history/` – Query historical scans.

This repository currently provides the core engine and infrastructure definition; the
DRF views/serializers and Celery tasks can be added following your organization’s
standards.

---

## 5. Using the Engine in Python

You can integrate the engine directly into any Python service (including Django
views, Celery tasks, or batch scripts).

### 5.1 Feature Extraction Only

```python
from engine.extractor import FeatureExtractor

extractor = FeatureExtractor()
result = extractor.extract(
    url="https://example.com/login?next=http://attacker.com",
    html=None,  # or DOM HTML from Playwright
)

print(result.feature_names)
print(result.vector)
print(result.metadata)
```

### 5.2 Graph‑Based Prediction with GNN + GT

```python
from engine.graph import GraphBuilder, HeteroGTConfig, HeteroGTModel
import torch

builder = GraphBuilder()
data = builder.build_subgraph("https://example.com/login?next=http://attacker.com")

config = HeteroGTConfig()
model = HeteroGTModel(config=config)

model.eval()
with torch.no_grad():
    probabilities, heatmap = model(data)

print("Phish probability:", float(probabilities[0]))

for domain, score in zip(data["domain"].name, heatmap["domain"].tolist()):
    print("Domain:", domain, "score:", score)
```

---

## 6. Training the GNN Model (Conceptual)

The current repository focuses on inference‑time graph construction and model
definition. A full training pipeline for `HeteroGTModel` would typically:

1. Use `datasets/build_training_dataset.py` to build a cleaned URL dataset with:
   - `url`
   - `label` (0 = benign, 1 = phishing)
2. For each URL, build a 1‑hop graph using `GraphBuilder`.
3. Batch subgraphs (or process them sequentially) through `HeteroGTModel`.
4. Optimize a binary cross‑entropy loss between:
   - Model probabilities for `url` nodes.
   - Ground truth labels.
5. Periodically evaluate on held‑out graphs and checkpoint model weights.

This training loop can be implemented using standard PyTorch practices with
PyTorch Geometric data loaders for heterogeneous graphs.

---

## 7. Next Steps

- Implement Django apps and DRF endpoints that:
  - Expose URL scan APIs.
  - Apply the cache‑first flow with Redis and PostgreSQL.
  - Log audit trails to support retraining.
- Implement Celery tasks for:
  - Deep WHOIS/DNS enrichment.
  - Playwright‑based DOM rendering and behavioral analysis.
- Integrate the engine with your SOC tooling (SIEM, SOAR, ticketing).

The current engine and blueprint are designed to be ready for those integrations
without major refactors.
