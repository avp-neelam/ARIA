# Appendix: Causal Validation of ARIA Belief Vectors

For the base ARIA model's saved final-phase weights (9 of 10 splits per dataset had `--save_weights` artifacts; split 0 was logged before that flag was enabled and is excluded), we test whether a node's belief weight on a channel predicts how much that channel's information actually mattered to its prediction. For channel c in {2-hop, global}, we zero out channel c's representation at every ARIA layer (holding the model's own gate beta -- computed from the un-ablated forward pass -- fixed) and measure each test node's drop in true-class probability, delta_c(i). We correlate beta_{i,c} against delta_c(i) across test nodes, per split, then average over the 9 final splits. Significance is a two-sided node-shuffle permutation test (1000 shuffles) on the Pearson r, per split.

| Dataset | 2-hop mean r | 2-hop top-vs-bottom-quartile Δ | 2-hop splits sig. (p<0.05) | Global mean r | Global top-vs-bottom-quartile Δ | Global splits sig. (p<0.05) |
|---|---|---|---|---|---|---|
| Cora | +0.017±0.068 | +0.0327 vs +0.0244 | 1/9 | -0.042±0.063 | +0.0019 vs +0.0168 | 0/9 |
| CiteSeer | +0.078±0.082 | +0.0228 vs +0.0130 | 4/9 | +0.014±0.043 | +0.0034 vs +0.0014 | 0/9 |
| PubMed | +0.029±0.162 | +0.0248 vs +0.0068 | 4/9 | +0.062±0.126 | +0.0202 vs +0.0074 | 4/9 |
| Texas | +0.017±0.138 | -0.0054 vs +0.0009 | 0/9 | -0.028±0.118 | +0.0018 vs -0.0007 | 0/9 |
| Wisconsin | +0.003±0.212 | +0.0033 vs -0.0008 | 1/9 | -0.049±0.159 | -0.0104 vs +0.0022 | 0/9 |
| Cornell | +0.074±0.233 | +0.0122 vs +0.0001 | 1/9 | -0.106±0.100 | -0.0030 vs +0.0116 | 0/9 |
| Amazon-Photo | +0.140±0.147 | +0.0470 vs +0.0193 | 7/9 | -0.032±0.146 | +0.0437 vs +0.0568 | 2/9 |
| Actor | +0.010±0.077 | +0.0004 vs -0.0005 | 2/9 | +0.010±0.020 | -0.0007 vs -0.0016 | 0/9 |
| USA-Airports | +0.289±0.363 | +0.2094 vs +0.0304 | 7/9 | +0.005±0.071 | -0.0012 vs -0.0032 | 0/9 |
| Brazil-Airports | +0.327±0.389 | +0.2071 vs -0.0055 | 5/9 | -0.028±0.106 | -0.0101 vs -0.0036 | 0/9 |
| Europe-Airports | +0.264±0.279 | +0.0953 vs +0.0088 | 6/9 | +0.022±0.133 | +0.0015 vs +0.0010 | 1/9 |

**Pooled across all 11 datasets:** mean 2-hop r = +0.113 (range +0.003 to +0.327); mean global r = -0.016 (range -0.106 to +0.062).

**Reading:** the 2-hop hypothesis holds clearly on the three struc2vec Airports graphs (USA-Airports r=+0.29 (7/9 splits significant), Brazil-Airports r=+0.33 (5/9 splits significant), Europe-Airports r=+0.26 (6/9 splits significant)), where node features carry no per-node signal on their own and multi-hop structure is genuinely the only source of predictive information. Per-split correlations are noisy (these graphs have as few as ~131 nodes total, so a single split's test set is tiny) -- the sign is positive in most but not all splits (USA: 7/9 splits positive, Brazil: 7/9 splits positive, Europe: 7/9 splits positive) -- but the 9-split mean is clearly and consistently positive across all three datasets and reaches the per-split permutation-test significance threshold in a majority of them. It is weak-to-absent on the citation networks, WebKB graphs, and Actor, where features already carry most of the signal and 2-hop propagation is not the model's main lever. **The global-channel hypothesis is not supported by this test**: correlations are small and inconsistent in sign across all 11 datasets (pooled mean r close to zero), so we do not claim the global channel's belief weight is causally validated the way the 2-hop channel's is -- this is an honest negative result, not a framing choice. As a methodology sanity check (not itself a headline claim), the same procedure applied to the self channel on the heterophilic WebKB graphs -- where the paper's own belief-composition figure already shows self-reliance dominating -- recovers a clear positive correlation (Texas r=+0.212, Wisconsin r=+0.119, Cornell r=+0.115), confirming the method detects real effects when they are present rather than only returning noise.
