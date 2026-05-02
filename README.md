# Product-to-Product Recommender System for Amazon Grocery Products

A hybrid multi-source recommender system that predicts co-purchase relationships between grocery products using graph signals, text embeddings, and collaborative filtering.

---

## Project Overview

Given a query product from the Amazon Grocery & Gourmet Food dataset, this system identifies and ranks products a customer is likely to also buy. The prediction target is held-out `also_buy` edges — co-purchase product links.

Each product is modeled as a **node in a graph**; the task is **link prediction** (product-to-product), not user-personalized retrieval.

---

## Dataset

| Metric | Value |
|---|---|
| Products (nodes) | 11,295 (largest connected component) |
| `also_buy` edges | 122,614 |
| `also_view` edges | 36,271 |
| Average graph degree | 24.76 |
| Average product rating | 4.33 |
| Median review count | 142 |

**Sources:** Amazon Grocery & Gourmet Food review file + product metadata file  
**Features:** Title, brand, category, description, features, price, rank, reviewer IDs, ratings, review text, review timestamps

> The dataset has no purchase timestamps or per-user purchase logs, so the scope is product-to-product similarity rather than sequential or user-personalized recommendation.

---

## Pipeline

### 1. Data Cleaning & Preprocessing
- Aggregated multi-row review file into one product-level table
- Normalized text fields ie. parsed messy price and rank strings into numerics
- Restricted graph to largest connected component
- Median imputation for missing numeric fields; retained missingness indicator columns

### 2. Feature Engineering

| Group | Features |
|---|---|
| **Content** | Sentence-transformer embeddings (`metadata_blob` + `review_text_blob`), category depth |
| **Quality and Popularity** | avg rating, review count, verified purchase ratio, avg review length, log-rank proxy |
| **Review-CF** | Jaccard-like reviewer overlap scores → 1,820,243 scored pairs |
| **Graph Structure** | Graph degree, SVD graph embeddings from adjacency matrix |

### 3. Train, Validation and Test Split

| Split | Edges |
|---|---|
| Train | 85,829 |
| Validation | 18,392 |
| Test | 18,393 |

**Leakage control:** 2,765 test `also_buy` edges overlapped with `also_view` before filtering → reduced to **0** after strict removal. Filtered `also_view` (30,665 edges) used only as auxiliary GNN message-passing context.

### 4. Models

#### Baselines
| Model | HitRate@10 | MRR@10 |
|---|---|---|
| Popularity | 0.074 | 0.026 |
| Matrix Factorization (SVD) | 0.361 | 0.156 |
| Content Text Similarity | 0.455 | 0.247 |

#### Graph Neural Networks
Trained with mixed negative sampling (random non-edges + `also_view`-only hard negatives).

| Model | Notes |
|---|---|
| **GraphSAGE** | Val AUC ~0.60; limited without node features |
| **LightGCN** | Val AUC ~0.48 (below random); no node features |
| **GAT** (+ text & metadata features) | Early stop epoch 7, val AUC 0.777 — best GNN |

#### Hybrid Multi-Source System (Stage 1 + Stage 2)

**Stage 1 — Candidate Retrieval (10 sources):**
Review-CF · `also_view` neighbors · GAT embeddings · GraphSAGE embeddings · LightGCN embeddings · metadata similarity · review text similarity · combined content similarity · SVD embeddings · 1-hop & 2-hop graph expansion

**Stage 2 — Reranking & Diversity:**
- Logistic reranker on pairwise features (brand/category match, title overlap, shared neighbors, price/rating/popularity diffs)
- Diversity post-processing to suppress same-brand / near-duplicate clusters

---

## Results (Test Set, K=10)

| Model | HitRate@10 | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| **Val_Tuned_Stage1_MultiSource** | **0.5883** | **0.2983** | 0.2242 | 0.2889 |
| Val_Tuned_RRF_Ensemble | 0.5859 | 0.2906 | **0.2250** | **0.3003** |
| Val_Tuned_TwoStage_Hybrid | 0.4860 | 0.2086 | 0.1602 | 0.2294 |
| Content_Text_Similarity | 0.4547 | 0.2123 | 0.1732 | 0.2474 |
| GAT_Text_Metadata_Graph | 0.4438 | 0.2023 | 0.1548 | 0.2123 |
| Val_Tuned_ReviewCF_AlsoView | 0.3815 | 0.1562 | 0.1415 | 0.2258 |
| Matrix_Factorization_SVD | 0.3606 | 0.1432 | 0.1085 | 0.1558 |
| Popularity | 0.0779 | 0.0233 | 0.0165 | 0.0260 |
| LightGCN | 0.0228 | 0.0106 | 0.0074 | 0.0096 |
| GraphSAGE | 0.0161 | 0.0052 | 0.0039 | 0.0057 |

> The hybrid multi-source model achieves **~8× improvement** in HitRate@10 over the popularity baseline.

### Cold-Start Evaluation (Low-degree products, K=10)

| Model | HitRate@10 | Recall@10 |
|---|---|---|
| Content_Text_Similarity | 0.464 | 0.412 |
| Val_Tuned_Stage1_MultiSource | 0.496 | 0.436 |
| Val_Tuned_RRF_Ensemble | 0.496 | 0.433 |
| GAT_Text_Metadata_Graph | 0.196 | 0.172 |

For sparse-graph products, **text embeddings are the primary signal**; Review-CF adds value but is limited when co-reviewer counts are low.

---

## Key Takeaways

- **Multi-source retrieval wins** — combining 10 signals substantially outperforms any single method
- **Node features are essential for GNNs** — GAT with text/metadata far outperformed feature-free LightGCN and GraphSAGE
- **Content features drive cold-start robustness** — sentence-transformer embeddings are reliable even for low-degree products
- **Leakage control matters** — strict `also_view`/`also_buy` overlap removal ensures evaluation validity
- **Diversity post-processing is necessary** — without it, recommendations cluster into near-identical brand variants

---

## Future Work

- Fine-tune sentence-transformer encoder on domain-specific grocery text
- Explore temporal graph models if purchase timestamps become available
- Replace logistic reranker with a neural reranker
- Extend Review-CF to full user-personalized retrieval with purchase history data

---

## Author

**Pradnya Tendolkar** — G45312425  
Applied Machine Learning — Final Project
