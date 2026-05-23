\# Critical Slowing Down Patch Warning



This repository contains the experimental code for patch-based early warning of critical slowing down in spatial SEIR epidemic systems.



\## Project Overview



The goal of this project is to predict the risk of approaching a critical transition in a spatial epidemic system by combining spatial patch information and temporal early-warning indicators.



The current version includes:



\- Spatial SEIR simulation code

\- Deep learning risk prediction model

\- Patch-based feature extraction

\- Baseline machine learning models

\- Final evaluation and visualization scripts

\- Ablation settings for full, image-only, and patch-only inputs



\## Repository Structure



```text

critical-slowing-patch-warning/

├── src/

│   └── seir\_model.py

├── scripts/

│   ├── train\_deep\_model.py

│   ├── train\_patch\_baselines.py

│   └── make\_final\_figures.py

├── archive/

│   └── v0\_legacy/

├── .gitignore

├── requirements.txt

└── README.md

```



\## Current Version



\- `v0.1-current-working-version`: original working version

\- `v0.2-research-repo-structure`: reorganized repository structure

\- `v0.3-docs-requirements`: documentation and reproducibility preparation



\## Main Results So Far



The current model achieves stable early-warning performance on simulated spatial SEIR data. Across different random seeds and feature modes, the model shows:



\- AUC around 0.86--0.91

\- AUPRC around 0.82--0.89

\- F1 around 0.73--0.81

\- Mean successful warning lead time around 14 steps

\- Median successful warning lead time around 13--14 steps



These results suggest that the model can provide useful early-warning signals before the system approaches a critical transition.



\## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the deep learning model:



```bash

python scripts/train\_deep\_model.py

```



Run baseline models:



```bash

python scripts/train\_patch\_baselines.py

```



Generate final figures:



```bash

python scripts/make\_final\_figures.py

```



\## Notes



Generated datasets, model weights, and result figures are not tracked by Git. They are ignored through `.gitignore`.



This repository is still under active development for manuscript preparation.

