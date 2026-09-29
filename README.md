# Lost in Serialization: How PDF Parsing Strategy Affects RAG Retrieval Quality

An empirical study of how PDF-to-text conversion affects Retrieval-Augmented Generation (RAG) retrieval. It compares naive text extraction (PyMuPDF) with structure-aware parsing (Docling) on 30 machine learning papers, using 88 gold-annotated questions covering text, table and equation evidence.

This is a pilot-scale study. See [Limitations](#limitations).

## Key findings

- **Tables:** structured parsing sharply reduces table-evidence retrieval. MRR falls from 0.450–0.592 (naive) to 0.091–0.196 (structured) across BM25, dense and hybrid retrieval. Paired Wilcoxon signed-rank tests give p = 0.000081–0.000162 with effect size r = 0.923–0.934, significant after Bonferroni correction (α ≈ 0.0056).
- **Text and equations:** no statistically significant difference between parsing strategies (text p = 0.445–0.970; equation p = 0.381–0.795).
- **Retrieval-invisible corruption:** in six confirmed cases, structured parsing replaced a paper's core equation with a `<!-- formula-not-decoded -->` placeholder. In three of the six, at least one retriever still ranked the corrupted chunk first. Rank-based metrics such as MRR and Hit@k cannot detect this failure.

## Corpus

30 arXiv papers in six groups of five. Two papers (VAE, Adam) contain no numerical results table, so they are evaluated on text and equation evidence only. This gives 88 questions: 30 text, 28 table, 30 equation.

| ID | Paper | arXiv | ID | Paper | arXiv |
|---|---|---|---|---|---|
| P001 | XGBoost | 1603.02754 | P016 | PPO | 1707.06347 |
| P002 | Mamba | 2312.00752 | P017 | DQN | 1312.5602 |
| P003 | DINOv2 | 2304.07193 | P018 | TRPO | 1502.05477 |
| P004 | LoRA | 2106.09685 | P019 | SAC | 1801.01290 |
| P005 | DreamerV3 | 2301.04104 | P020 | Rainbow | 1710.02298 |
| P006 | Attention Is All You Need | 1706.03762 | P021 | GAN | 1406.2661 |
| P007 | BERT | 1810.04805 | P022 | VAE | 1312.6114 |
| P008 | GPT-3 | 2005.14165 | P023 | DDPM | 2006.11239 |
| P009 | T5 | 1910.10683 | P024 | Latent Diffusion | 2112.10752 |
| P010 | LLaMA | 2302.13971 | P025 | VQ-VAE | 1711.00937 |
| P011 | ResNet | 1512.03385 | P026 | Adam | 1412.6980 |
| P012 | ViT | 2010.11929 | P027 | Batch Normalization | 1502.03167 |
| P013 | CLIP | 2103.00020 | P028 | Layer Normalization | 1607.06450 |
| P014 | EfficientNet | 1905.11946 | P029 | FlashAttention | 2205.14135 |
| P015 | SimCLR | 2002.05709 | P030 | Chinchilla | 2203.15556 |

## Method

- **Naive extraction:** PyMuPDF (`page.get_text()`), pages concatenated in order.
- **Structured extraction:** Docling, exported as Markdown with tables in Markdown syntax.
- **Chunking:** shared word-budget chunker with a 200-word target. Chunks under 60 words are merged. Tables over 600 words are split at row boundaries with the header repeated. The structured chunker also splits on headings and table boundaries. Result: 2,051 naive chunks and 2,561 structured chunks.
- **Retrieval:**
  - BM25 (`rank_bm25`, Okapi, lowercased whitespace tokens)
  - dense (`BAAI/bge-small-en-v1.5`, L2-normalised, dot product, with the BGE query instruction prefix)
  - hybrid (Reciprocal Rank Fusion, k = 60)
- **Metrics:** MRR and Hit@1/3/5.
- **Statistics:** paired Wilcoxon signed-rank test per evidence type and retriever (nine primary comparisons, Bonferroni α ≈ 0.0056). Effect size is the matched-pairs rank-biserial correlation. Aggregate tests are secondary.
- **Gold annotation:** every question has verified gold chunk IDs in both corpora, found by keyword-assisted search and checked by reading the full chunk text against the source paper.

## Results

Aggregate retrieval performance (88 questions):

| Corpus | Retriever | MRR | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|
| Naive | BM25 | 0.466 | 0.364 | 0.557 | 0.580 |
| Naive | Dense | 0.460 | 0.352 | 0.500 | 0.557 |
| Naive | Hybrid | 0.466 | 0.330 | 0.557 | 0.614 |
| Structured | BM25 | 0.319 | 0.239 | 0.364 | 0.375 |
| Structured | Dense | 0.317 | 0.216 | 0.341 | 0.420 |
| Structured | Hybrid | 0.337 | 0.239 | 0.307 | 0.511 |

Confirmed `formula-not-decoded` cases, with the rank of the gold chunk in the structured corpus:

| Paper | Formula | BM25 | Dense | Hybrid |
|---|---|---|---|---|
| ResNet | Residual equation (Eq. 1) | 1 | 1 | 1 |
| ViT | Self-attention, Eqs. 5–8 | 30 | 1 | 1 |
| SimCLR | NT-Xent loss | 2 | 4 | 2 |
| PPO | Clipped objective L^CLIP | 802 | 7 | 30 |
| LayerNorm | LN(z; α, β) | 11 | 1 | 4 |
| FlashAttention | HBM-access bound (Thm. 2) | 13 | 8 | 9 |

Per-evidence-type results are in `data/summary_by_evidence_type.csv` and `data/summary_overall.csv`.

## Repository structure

```
.
├── data/
│   ├── raw_pdfs/                    # source PDFs, P001–P030
│   ├── parsed_naive/                # PyMuPDF output (.txt)
│   ├── parsed_structured/           # Docling output (.md)
│   ├── chunks_naive/chunks.json
│   ├── chunks_structured/chunks.json
│   ├── summary_overall.csv
│   └── summary_by_evidence_type.csv
├── questions/
│   └── questions.json               # 88 questions with gold chunk IDs
├── results/
│   └── evaluation_results.csv       # 528 rows: 88 questions x 2 corpora x 3 retrievers
├── src/
│   ├── parsing/
│   │   ├── naive_extract.py         # PyMuPDF extraction
│   │   └── structured_extract.py    # Docling extraction
│   ├── retrieval/
│   │   ├── chunk_documents.py       # shared chunker
│   │   ├── bm25_search.py
│   │   └── dense_search.py
│   └── eval/
│       ├── eval_full.py             # BM25 + dense + hybrid over both corpora
│       ├── aggregate_results.py     # summary tables
│       ├── wilcoxon_test.py         # aggregate significance tests
│       ├── wilcoxon_by_evidence_type.py
│       ├── find_gold_chunk.py       # gold-chunk search helpers
│       ├── find_all_gold_chunks.py
│       ├── verify_original_gold_chunks.py
│       ├── find_size_outliers.py
│       └── inspect_chunks.py
├── requirements.txt
└── README.md
```

## Data formats

- `questions/questions.json`: one object per question, with `paper_id`, `question_id`, `evidence_type` (`text`, `table` or `equation`), `question`, `gold_answer`, `source_location`, `gold_chunk_ids_naive` and `gold_chunk_ids_structured`.
- `chunks.json`: one object per chunk, with `chunk_id`, `paper_id`, `chunk_index`, `text`, `section` and `chunk_type`.
- `evaluation_results.csv`: `question_id`, `paper_id`, `evidence_type`, `corpus_condition`, `retriever`, `rank`, `reciprocal_rank`, `hit_at_1`, `hit_at_3`, `hit_at_5`.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Extract text
python src/parsing/naive_extract.py
python src/parsing/structured_extract.py

# 2. Chunk both corpora
python src/retrieval/chunk_documents.py

# 3. Run the evaluation (writes results/evaluation_results.csv)
python src/eval/eval_full.py

# 4. Summaries and significance tests
python src/eval/aggregate_results.py
python src/eval/Wilcoxon_test.py
python src/eval/wilcoxon_by_evidence_type.py
```

Run all commands from the repository root. The first run downloads the embedding model.

## Limitations

- **Scale:** 30 papers and 88 questions, all machine learning papers from arXiv. Results may not generalise to other document types.
- **Annotation:** a single annotator with AI-assisted search and verification, and no inter-annotator agreement. Keyword-assisted evidence discovery may introduce lexical-selection bias.
- **Parsing and chunking are coupled:** the structured pipeline's chunker also uses headings and table boundaries, and the corpora differ in chunk count. The reported differences reflect parsing plus the chunking it enables.
- **Scope of comparison:** one naive extractor, one structure-aware parser and one embedding model. Retrieval only; downstream answer quality is not evaluated.
- **Statistics:** questions come from 30 papers, so question-level observations may be clustered within papers. Non-significant results mean no difference was detected, not that the strategies are equivalent.
- **Corruption cases:** the six `formula-not-decoded` cases were identified during annotation. They are not a systematic estimate of how often equations are lost.