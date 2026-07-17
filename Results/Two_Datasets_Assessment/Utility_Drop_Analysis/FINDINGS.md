# Utility Drop Analysis — Key Findings

**Primary metric:** Accuracy (Mean ± SD across seeds)
**Focus datasets:** Cancer, Mushroom
**Multi-level leakage data present:** No (Leakage=0% only; drop = TRTR−TSTR)

## Highest utility generator
1. **ForestDiffusion** — mean Accuracy 0.863
2. WGAN_GP — 0.802

## Best classifier
1. **Random Forest** — mean Accuracy 0.701
2. Extra Trees — 0.696

## Most robust generator (lowest utility loss)
1. **ForestDiffusion** — mean loss 0.034

## Classifier least affected by drop
1. **Naive Bayes** — mean loss 0.092
