| Method | Features | Acc in domain | AUC in domain | Acc out of domain | AUC out of domain | Acc pooled | AUC pooled | AUC pooled, concatenated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mass-mean, activations | 35,840 | 74.2 | 84.4 | 82.2 | 89.4 | 77.4 | 86.3 | 82.8 |
| Logistic regression, activations | 35,840 | 75.2 | 82.8 | 80.5 | 86.4 | 77.2 | 84.2 | 82.2 |
| Logistic regression, activations (PCA 300) | 300 | 72.1 | 81.5 | 84.2 | 87.9 | 76.8 | 84.0 | 83.0 |
| Logistic regression, CI of untrained decomposition | 960 | 59.1 | 64.6 | 52.2 | 62.1 | 56.4 | 63.6 | 59.0 |
| Logistic regression, CI (tpd-capitals-true) | 960 | 61.2 | 65.8 | 55.0 | 84.5 | 58.8 | 73.1 | 65.8 |
| Random direction, CI (tpd-capitals-true) | 960 | – | 49.4 | – | 46.1 | – | 48.1 | 49.8 |
| Logistic regression, CI top 10 (tpd-capitals-true) | 10 | 56.1 | 61.3 | 51.8 | 80.3 | 54.4 | 68.7 | 60.7 |
| Logistic regression, CI random 10 (tpd-capitals-true) | 10 | 56.7 | 58.8 | 57.8 | 72.6 | 57.1 | 64.2 | 59.7 |
| Logistic regression, CI top 40 (tpd-capitals-true) | 40 | 58.5 | 64.4 | 69.1 | 80.9 | 62.6 | 70.8 | 67.8 |
| Logistic regression, CI random 40 (tpd-capitals-true) | 40 | 57.5 | 60.7 | 65.7 | 73.6 | 60.7 | 65.7 | 65.2 |
| Logistic regression, CI top 80 (tpd-capitals-true) | 80 | 57.6 | 66.0 | 68.8 | 80.1 | 61.9 | 71.5 | 67.5 |
| Logistic regression, CI random 80 (tpd-capitals-true) | 80 | 59.0 | 62.2 | 62.2 | 77.1 | 60.2 | 68.0 | 64.8 |
| Logistic regression, CI top 160 (tpd-capitals-true) | 160 | 57.0 | 66.9 | 67.5 | 81.4 | 61.1 | 72.5 | 66.5 |
| Logistic regression, CI random 160 (tpd-capitals-true) | 160 | 60.4 | 62.9 | 60.0 | 77.4 | 60.2 | 68.5 | 65.3 |
| Logistic regression, CI top 8 per layer (tpd-capitals-true) | 80 | 60.6 | 61.1 | 67.8 | 73.2 | 63.4 | 65.8 | 66.8 |
| Logistic regression, CI PCA 10 (tpd-capitals-true) | 10 | 59.4 | 64.6 | 68.9 | 87.5 | 63.1 | 73.5 | 70.3 |
| Logistic regression, CI PCA 50 (tpd-capitals-true) | 50 | 62.4 | 69.1 | 76.2 | 84.7 | 67.8 | 75.2 | 74.5 |
| Logistic regression, CI (tpd-capitals-false) | 960 | 63.0 | 69.5 | 73.7 | 82.6 | 67.2 | 74.6 | 72.2 |
| Logistic regression, CI top 10 (tpd-capitals-false) | 10 | 67.0 | 73.3 | 65.4 | 83.5 | 66.4 | 77.3 | 72.2 |
| Logistic regression, CI random 10 (tpd-capitals-false) | 10 | 61.8 | 67.4 | 70.5 | 77.1 | 65.2 | 71.2 | 65.7 |
| Logistic regression, CI top 40 (tpd-capitals-false) | 40 | 63.6 | 70.2 | 68.8 | 84.7 | 65.6 | 75.8 | 71.4 |
| Logistic regression, CI random 40 (tpd-capitals-false) | 40 | 63.6 | 68.5 | 73.4 | 82.1 | 67.4 | 73.8 | 71.7 |
| Logistic regression, CI top 80 (tpd-capitals-false) | 80 | 63.3 | 68.2 | 76.4 | 84.9 | 68.4 | 74.7 | 73.7 |
| Logistic regression, CI random 80 (tpd-capitals-false) | 80 | 63.3 | 67.9 | 75.7 | 85.4 | 68.2 | 74.7 | 72.3 |
| Logistic regression, CI top 160 (tpd-capitals-false) | 160 | 66.4 | 69.5 | 72.7 | 84.3 | 68.8 | 75.3 | 72.4 |
| Logistic regression, CI random 160 (tpd-capitals-false) | 160 | 63.3 | 67.9 | 75.9 | 85.2 | 68.2 | 74.6 | 71.2 |
| Logistic regression, CI top 8 per layer (tpd-capitals-false) | 80 | 63.9 | 67.4 | 75.0 | 85.1 | 68.3 | 74.3 | 73.5 |
| Logistic regression, CI PCA 10 (tpd-capitals-false) | 10 | 63.3 | 68.0 | 75.9 | 84.2 | 68.2 | 74.3 | 73.6 |
| Logistic regression, CI PCA 50 (tpd-capitals-false) | 50 | 66.1 | 73.1 | 74.6 | 86.6 | 69.4 | 78.4 | 75.2 |
| Logistic regression, CI (tpd-liu-true) | 960 | 66.4 | 73.8 | 52.8 | 68.5 | 61.1 | 71.7 | 66.6 |
| Logistic regression, CI top 10 (tpd-liu-true) | 10 | 45.5 | 49.6 | 52.9 | 69.0 | 48.3 | 57.1 | 52.3 |
| Logistic regression, CI random 10 (tpd-liu-true) | 10 | 56.8 | 64.5 | 50.2 | 52.6 | 54.3 | 59.9 | 54.9 |
| Logistic regression, CI top 40 (tpd-liu-true) | 40 | 59.4 | 67.8 | 63.9 | 72.1 | 61.1 | 69.4 | 61.8 |
| Logistic regression, CI random 40 (tpd-liu-true) | 40 | 59.3 | 67.6 | 51.3 | 64.3 | 56.2 | 66.3 | 59.7 |
| Logistic regression, CI top 80 (tpd-liu-true) | 80 | 60.0 | 68.5 | 61.5 | 72.0 | 60.6 | 69.9 | 62.8 |
| Logistic regression, CI random 80 (tpd-liu-true) | 80 | 63.3 | 71.9 | 52.5 | 61.3 | 59.1 | 67.8 | 62.7 |
| Logistic regression, CI top 160 (tpd-liu-true) | 160 | 59.7 | 68.4 | 53.8 | 67.7 | 57.4 | 68.1 | 62.8 |
| Logistic regression, CI random 160 (tpd-liu-true) | 160 | 60.7 | 71.0 | 51.5 | 67.6 | 57.1 | 69.6 | 61.4 |
| Logistic regression, CI top 8 per layer (tpd-liu-true) | 80 | 60.0 | 68.0 | 55.5 | 72.1 | 58.3 | 69.6 | 62.6 |
| Logistic regression, CI PCA 10 (tpd-liu-true) | 10 | 67.6 | 77.8 | 49.4 | 67.3 | 60.5 | 73.8 | 64.0 |
| Logistic regression, CI PCA 50 (tpd-liu-true) | 50 | 64.2 | 70.7 | 53.1 | 63.4 | 59.9 | 67.8 | 64.1 |
| Logistic regression, CI (tpd-liu-false) | 960 | 73.9 | 82.4 | 83.1 | 84.9 | 77.5 | 83.4 | 80.6 |
| Logistic regression, CI top 10 (tpd-liu-false) | 10 | 66.1 | 70.5 | 81.9 | 87.9 | 72.2 | 77.3 | 76.0 |
| Logistic regression, CI random 10 (tpd-liu-false) | 10 | 56.4 | 61.3 | 49.5 | 50.6 | 53.7 | 57.1 | 52.6 |
| Logistic regression, CI top 40 (tpd-liu-false) | 40 | 63.9 | 66.6 | 78.1 | 83.9 | 69.5 | 73.3 | 71.5 |
| Logistic regression, CI random 40 (tpd-liu-false) | 40 | 61.7 | 66.1 | 55.8 | 57.8 | 59.4 | 62.9 | 57.5 |
| Logistic regression, CI top 80 (tpd-liu-false) | 80 | 70.3 | 74.7 | 81.1 | 87.2 | 74.5 | 79.6 | 75.1 |
| Logistic regression, CI random 80 (tpd-liu-false) | 80 | 65.0 | 72.4 | 63.8 | 66.0 | 64.5 | 69.9 | 64.4 |
| Logistic regression, CI top 160 (tpd-liu-false) | 160 | 69.7 | 76.1 | 83.1 | 86.7 | 74.9 | 80.2 | 77.5 |
| Logistic regression, CI random 160 (tpd-liu-false) | 160 | 66.1 | 71.8 | 60.8 | 69.1 | 64.0 | 70.8 | 64.2 |
| Logistic regression, CI top 8 per layer (tpd-liu-false) | 80 | 62.7 | 70.9 | 78.7 | 87.2 | 68.9 | 77.3 | 69.4 |
| Logistic regression, CI PCA 10 (tpd-liu-false) | 10 | 68.5 | 76.0 | 66.8 | 86.3 | 67.8 | 80.0 | 67.6 |
| Logistic regression, CI PCA 50 (tpd-liu-false) | 50 | 69.1 | 77.4 | 72.7 | 86.6 | 70.5 | 81.0 | 68.1 |
| Logistic regression, CI (tpd-tqa-true) | 960 | 53.9 | 52.3 | 63.8 | 66.8 | 57.8 | 58.0 | 59.8 |
| Logistic regression, CI top 10 (tpd-tqa-true) | 10 | 53.9 | 55.6 | 50.7 | 61.6 | 52.7 | 57.9 | 51.9 |
| Logistic regression, CI random 10 (tpd-tqa-true) | 10 | 50.8 | 50.8 | 53.8 | 57.8 | 52.0 | 53.5 | 51.0 |
| Logistic regression, CI top 40 (tpd-tqa-true) | 40 | 58.2 | 56.1 | 58.1 | 60.3 | 58.2 | 57.7 | 55.3 |
| Logistic regression, CI random 40 (tpd-tqa-true) | 40 | 50.7 | 49.8 | 53.7 | 59.7 | 51.8 | 53.6 | 52.1 |
| Logistic regression, CI top 80 (tpd-tqa-true) | 80 | 60.6 | 58.3 | 60.9 | 67.4 | 60.7 | 61.8 | 59.6 |
| Logistic regression, CI random 80 (tpd-tqa-true) | 80 | 54.4 | 56.1 | 63.7 | 68.4 | 58.0 | 60.9 | 60.1 |
| Logistic regression, CI top 160 (tpd-tqa-true) | 160 | 54.8 | 52.2 | 62.2 | 68.4 | 57.7 | 58.5 | 59.9 |
| Logistic regression, CI random 160 (tpd-tqa-true) | 160 | 54.7 | 54.5 | 64.7 | 67.7 | 58.6 | 59.6 | 59.7 |
| Logistic regression, CI top 8 per layer (tpd-tqa-true) | 80 | 57.3 | 58.9 | 66.1 | 72.3 | 60.7 | 64.1 | 61.8 |
| Logistic regression, CI PCA 10 (tpd-tqa-true) | 10 | 54.8 | 56.2 | 49.4 | 49.5 | 52.7 | 53.6 | 52.7 |
| Logistic regression, CI PCA 50 (tpd-tqa-true) | 50 | 57.0 | 64.2 | 57.2 | 60.3 | 57.1 | 62.7 | 58.8 |
| Logistic regression, CI (tpd-tqa-false) | 960 | 62.7 | 67.6 | 78.3 | 85.0 | 68.8 | 74.4 | 69.6 |
| Logistic regression, CI top 10 (tpd-tqa-false) | 10 | 64.2 | 70.9 | 74.3 | 87.5 | 68.2 | 77.3 | 69.5 |
| Logistic regression, CI random 10 (tpd-tqa-false) | 10 | 48.6 | 47.5 | 51.8 | 52.2 | 49.8 | 49.3 | 47.9 |
| Logistic regression, CI top 40 (tpd-tqa-false) | 40 | 63.6 | 70.2 | 73.9 | 87.6 | 67.6 | 77.0 | 66.5 |
| Logistic regression, CI random 40 (tpd-tqa-false) | 40 | 51.8 | 52.5 | 56.4 | 62.1 | 53.6 | 56.2 | 53.5 |
| Logistic regression, CI top 80 (tpd-tqa-false) | 80 | 62.1 | 66.0 | 76.7 | 85.1 | 67.8 | 73.4 | 70.1 |
| Logistic regression, CI random 80 (tpd-tqa-false) | 80 | 62.5 | 65.0 | 64.9 | 78.2 | 63.4 | 70.2 | 63.7 |
| Logistic regression, CI top 160 (tpd-tqa-false) | 160 | 63.9 | 70.0 | 78.3 | 85.0 | 69.5 | 75.8 | 67.9 |
| Logistic regression, CI random 160 (tpd-tqa-false) | 160 | 63.3 | 68.5 | 74.6 | 85.1 | 67.7 | 75.0 | 68.2 |
| Logistic regression, CI top 8 per layer (tpd-tqa-false) | 80 | 63.3 | 67.1 | 76.9 | 84.9 | 68.6 | 74.0 | 67.2 |
| Logistic regression, CI PCA 10 (tpd-tqa-false) | 10 | 67.3 | 73.1 | 70.8 | 89.2 | 68.6 | 79.4 | 73.4 |
| Logistic regression, CI PCA 50 (tpd-tqa-false) | 50 | 65.8 | 72.4 | 68.8 | 88.1 | 66.9 | 78.5 | 71.3 |
| Logistic regression, CI (tpd-capitals-liu-tqa-true) | 960 | 67.3 | 72.1 | 55.0 | 65.3 | 62.5 | 69.4 | 65.8 |
| Logistic regression, CI top 10 (tpd-capitals-liu-tqa-true) | 10 | 51.5 | 58.4 | 52.0 | 50.2 | 51.7 | 55.2 | 52.3 |
| Logistic regression, CI random 10 (tpd-capitals-liu-tqa-true) | 10 | 59.3 | 62.8 | 50.1 | 51.2 | 55.7 | 58.3 | 55.8 |
| Logistic regression, CI top 40 (tpd-capitals-liu-tqa-true) | 40 | 60.9 | 64.3 | 51.0 | 46.6 | 57.1 | 57.4 | 56.6 |
| Logistic regression, CI random 40 (tpd-capitals-liu-tqa-true) | 40 | 60.1 | 64.0 | 52.2 | 53.9 | 57.0 | 60.1 | 57.5 |
| Logistic regression, CI top 80 (tpd-capitals-liu-tqa-true) | 80 | 66.7 | 67.1 | 54.4 | 68.8 | 61.9 | 67.8 | 66.0 |
| Logistic regression, CI random 80 (tpd-capitals-liu-tqa-true) | 80 | 64.4 | 70.1 | 55.3 | 63.1 | 60.8 | 67.4 | 65.6 |
| Logistic regression, CI top 160 (tpd-capitals-liu-tqa-true) | 160 | 68.2 | 74.2 | 58.2 | 70.8 | 64.3 | 72.8 | 69.1 |
| Logistic regression, CI random 160 (tpd-capitals-liu-tqa-true) | 160 | 63.1 | 67.6 | 53.4 | 57.2 | 59.3 | 63.6 | 62.0 |
| Logistic regression, CI top 8 per layer (tpd-capitals-liu-tqa-true) | 80 | 66.4 | 70.5 | 53.3 | 64.1 | 61.3 | 68.0 | 64.3 |
| Logistic regression, CI PCA 10 (tpd-capitals-liu-tqa-true) | 10 | 62.1 | 70.0 | 53.7 | 62.7 | 58.8 | 67.2 | 62.7 |
| Logistic regression, CI PCA 50 (tpd-capitals-liu-tqa-true) | 50 | 63.3 | 69.1 | 52.7 | 62.2 | 59.2 | 66.4 | 61.7 |
| Logistic regression, CI (tpd-capitals-liu-tqa-false) | 960 | 70.3 | 78.7 | 78.6 | 85.2 | 73.5 | 81.3 | 76.5 |
| Logistic regression, CI top 10 (tpd-capitals-liu-tqa-false) | 10 | 65.2 | 74.2 | 77.0 | 87.2 | 69.8 | 79.3 | 76.6 |
| Logistic regression, CI random 10 (tpd-capitals-liu-tqa-false) | 10 | 58.0 | 62.5 | 53.2 | 53.6 | 56.1 | 59.1 | 54.9 |
| Logistic regression, CI top 40 (tpd-capitals-liu-tqa-false) | 40 | 68.2 | 75.3 | 79.0 | 83.9 | 72.4 | 78.6 | 75.1 |
| Logistic regression, CI random 40 (tpd-capitals-liu-tqa-false) | 40 | 64.1 | 69.8 | 64.0 | 67.2 | 64.1 | 68.8 | 63.8 |
| Logistic regression, CI top 80 (tpd-capitals-liu-tqa-false) | 80 | 67.6 | 76.0 | 80.3 | 85.3 | 72.5 | 79.6 | 75.9 |
| Logistic regression, CI random 80 (tpd-capitals-liu-tqa-false) | 80 | 64.9 | 71.5 | 66.7 | 72.5 | 65.6 | 71.9 | 67.7 |
| Logistic regression, CI top 160 (tpd-capitals-liu-tqa-false) | 160 | 68.2 | 78.6 | 80.3 | 87.2 | 72.9 | 81.9 | 77.3 |
| Logistic regression, CI random 160 (tpd-capitals-liu-tqa-false) | 160 | 66.5 | 72.9 | 70.1 | 75.5 | 67.9 | 73.9 | 68.2 |
| Logistic regression, CI top 8 per layer (tpd-capitals-liu-tqa-false) | 80 | 68.8 | 76.1 | 80.0 | 84.0 | 73.2 | 79.2 | 75.0 |
| Logistic regression, CI PCA 10 (tpd-capitals-liu-tqa-false) | 10 | 67.3 | 72.3 | 72.1 | 86.4 | 69.1 | 77.8 | 74.5 |
| Logistic regression, CI PCA 50 (tpd-capitals-liu-tqa-false) | 50 | 67.0 | 73.5 | 75.4 | 84.8 | 70.2 | 77.9 | 71.8 |

Means over held-out domains (leave-one-domain-out). In-domain statements are the target data of the liu and capitals-liu-tqa runs; out-of-domain ones were seen by no run. The concatenated AUC pools the held-out scores of all 18 domains into one ranking.

- in domain (11): animals, capitals, companies, counterfact, elements, facts, hotpot_qa_re, inventions, nq_re, sciq, triva_qa_re
- out of domain (7): got_larger_than, got_likely, got_neg_cities, got_neg_sp_en_trans, got_smaller_than, got_sp_en_trans, truthfulqa
- pooled (18): animals, capitals, companies, counterfact, elements, facts, got_larger_than, got_likely, got_neg_cities, got_neg_sp_en_trans, got_smaller_than, got_sp_en_trans, hotpot_qa_re, inventions, nq_re, sciq, triva_qa_re, truthfulqa
