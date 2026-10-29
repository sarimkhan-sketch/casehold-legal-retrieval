## Full-pool retrieval (test split)

| Ranking model | P@10 | Recall@10 | Recall@100 | MAP (= MRR) | nDCG@10 | Wilcoxon p vs BM25 |
|---|---|---|---|---|---|---|
| TF-IDF | 0.049 | 0.488 | 0.728 | 0.317 | 0.349 | 0.41 |
| BM25 | 0.049 | 0.492 | 0.709 | 0.320 | 0.354 | - |
| Sentence Transformer | 0.037 | 0.369 | 0.604 | 0.234 | 0.258 | 1e-65 |
| Hybrid RRF (TF-IDF + BM25 + Dense) | 0.053 | 0.526 | 0.762 | 0.327 | 0.367 | 3.2e-16 |
| Hybrid RRF, k = 5 | 0.054 | 0.535 | 0.759 | 0.341 | 0.380 | 7.8e-21 |
| Hybrid RRF (BM25 + Dense) | 0.049 | 0.488 | 0.747 | 0.314 | 0.347 | 0.63 |
| Hybrid + cross-encoder | 0.039 | 0.392 | 0.762 | 0.181 | 0.215 | 5e-66 |
| Dense, fine-tuned (ext.) | 0.058 | 0.581 | 0.802 | 0.395 | 0.432 | 1.9e-45 |
| Hybrid RRF (BM25 + fine-tuned Dense) | 0.060 | 0.596 | 0.821 | 0.401 | 0.440 | 4.3e-162 |

## 5-choice setting (original CaseHOLD format)

| Ranking model | Accuracy | MRR | Recall@2 | nDCG@5 |
|---|---|---|---|---|
| TF-IDF | 0.554 | 0.726 | 0.764 | 0.795 |
| BM25 | 0.579 | 0.745 | 0.788 | 0.809 |
| Sentence Transformer | 0.523 | 0.705 | 0.738 | 0.779 |
| Dense, fine-tuned (ext.) | 0.637 | 0.785 | 0.842 | 0.839 |
| Hybrid RRF (TF-IDF + BM25 + Dense) | 0.584 | 0.746 | 0.784 | 0.810 |
| Hybrid RRF (BM25 + Dense) | 0.563 | 0.736 | 0.787 | 0.803 |
| Hybrid RRF (BM25 + fine-tuned Dense) | 0.624 | 0.777 | 0.834 | 0.833 |
| Cross-encoder (ms-marco) | 0.386 | 0.605 | 0.611 | 0.703 |
