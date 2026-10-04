# Completion v3 fresh final test

Scores measure only the withheld sequence. Original observed bases are preserved. This is chromosome completion; plasmids and clinical antibiotic validation are outside this experiment.

|Species|Observed|Genomes|Missing recall|Missing precision|Missing F1|95/95 successes|No candidate|
|---|---:|---:|---:|---:|---:|---:|---:|
|ALL|20%|50|89.55%|89.91%|89.70%|6|0|
|ALL|65%|50|90.14%|89.06%|89.54%|6|0|
|ALL|90%|50|90.97%|90.88%|90.82%|15|0|
|ALL|95%|50|90.03%|91.60%|90.54%|24|0|
|ALL|98%|50|92.79%|93.43%|92.93%|34|0|
|ECOLI|20%|10|82.59%|85.18%|83.81%|0|0|
|ECOLI|65%|10|84.87%|84.49%|84.61%|0|0|
|ECOLI|90%|10|83.49%|87.80%|85.45%|0|0|
|ECOLI|95%|10|86.09%|91.62%|88.36%|2|0|
|ECOLI|98%|10|83.38%|83.86%|83.59%|3|0|
|KPNEU|20%|10|90.67%|89.68%|90.17%|2|0|
|KPNEU|65%|10|89.92%|89.37%|89.61%|2|0|
|KPNEU|90%|10|92.51%|91.63%|91.96%|4|0|
|KPNEU|95%|10|87.12%|86.06%|86.17%|4|0|
|KPNEU|98%|10|92.31%|96.39%|93.54%|6|0|
|SAUR|20%|10|92.99%|92.06%|92.52%|1|0|
|SAUR|65%|10|93.22%|90.88%|91.96%|1|0|
|SAUR|90%|10|91.98%|92.61%|92.21%|4|0|
|SAUR|95%|10|90.61%|91.43%|91.01%|6|0|
|SAUR|98%|10|95.20%|94.91%|95.05%|9|0|
|PAER|20%|10|91.17%|91.60%|91.38%|1|0|
|PAER|65%|10|93.36%|92.18%|92.69%|2|0|
|PAER|90%|10|90.31%|86.70%|88.39%|1|0|
|PAER|95%|10|88.66%|91.99%|89.93%|4|0|
|PAER|98%|10|94.43%|93.66%|94.00%|7|0|
|ABAU|20%|10|90.30%|91.03%|90.65%|2|0|
|ABAU|65%|10|89.35%|88.36%|88.80%|1|0|
|ABAU|90%|10|96.59%|95.67%|96.11%|6|0|
|ABAU|95%|10|97.68%|96.88%|97.25%|8|0|
|ABAU|98%|10|98.63%|98.33%|98.46%|9|0|

Early stops: 0/50. Reaching 100% input is fallback, not successful inference.
The confidence event requires both missing-base recovery and prediction precision >=95%; it does not mean exact DNA, preserved resistance mutations, or correct MIC.
