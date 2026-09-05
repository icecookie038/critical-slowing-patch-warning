# Published ecological CSD benchmark data

This project uses public data associated with four peer-reviewed studies that
reported direct recovery slowing or early-warning patterns consistent with
critical slowing down. Raw files are stored under `data/raw/` and are excluded
from Git.

## Datasets used

| Study | Ecological system | Design used here | Repository |
| --- | --- | --- | --- |
| van Belzen et al. (2017) | Tidal-marsh vegetation | Two sites; 8/11 irregular binary maps at 0.25 m; direct recovery events | [Zenodo 4998258](https://zenodo.org/records/4998258) |
| Clements & Ozgul (2016) | Predator–prey microcosms | 37 daily population trajectories; abundance and body-size traits | [Nature supplementary data](https://www.nature.com/articles/ncomms10984#Sec14) |
| Dai et al. (2015) | Cooperative yeast | 48 parallel populations over 21 daily states under two deteriorating drivers and controls | [Dryad dryad.k30v3](https://datadryad.org/dataset/doi:10.5061/dryad.k30v3) |
| Rindi et al. (2018) | Intertidal macroalgal canopy | Two years; 16 transects; 5 × 30 spatial observations per map | [Figshare 6200822.v2](https://figshare.com/articles/dataset/Experimental_evidence_of_spatial_signatures_of_approaching_regime_shifts_in_macroalgal_canopies/6200822) |

The Dai, Rindi, and tidal-marsh repository records are CC0. The Clements paper
and its supplementary material are distributed with the open-access article.

## Download and verification

Download the three v3.6 benchmark datasets with:

```bash
python scripts/download_published_csd_benchmarks.py
```

Download the tidal-marsh bundle separately with:

```bash
python scripts/download_tidal_marsh_data.py
```

Both downloaders use an atomic `.part` file and verify the downloaded content
before making it available to the analysis. Published repository MD5 values are
used where supplied; stable observed hashes are recorded for the Nature
supplementary files.

| Dataset/file | Expected MD5 |
| --- | --- |
| Clements `experimental_data.xls` | `15e7e71bb7cb8a7b34f118bf5907d876` |
| Clements bifurcation code | `177a6e152a0acb17e61e08ff2bd317bb` |
| Clements EWS code | `8362df038a2dc9d5644e77074ea25758` |
| Dai `data_deterioration.zip` | `f739beab74af2f981d410c10277ec364` |
| Rindi `cys_cover_avg_1yst.txt` | `d929fb705a7f5996b8415e7843ed1203` |
| Rindi `cys_cover_avg_2nd.txt` | `c1036005b61db6c1f217e28d623a22f5` |
| Rindi `dat_cl_1yst.txt` | `4549dd387a9f95ffdbf83dad868e000d` |
| Rindi `dat_cl_2nd.txt` | `a4e5f7dff18150bbd97cc14c0c4194af` |
| Rindi `Exp_data_2014.txt` | `f68a4627374dc1880bf4b649d1089323` |
| Rindi `Exp_data_2015.txt` | `d185a3a226ab43a4467370295b668058` |
| van Belzen data bundle | `eb545ba9e8e68e41c7f2988157e447c7` |

## Published sources

- Clements, C. F. & Ozgul, A. (2016). [Including trait-based early warning
  signals helps predict population collapse](https://doi.org/10.1038/ncomms10984).
  *Nature Communications*, 7, 10984.
- Dai, L., Korolev, K. S. & Gore, J. (2015). [Relation between stability and
  resilience determines the performance of early warning signals under
  different environmental drivers](https://doi.org/10.1073/pnas.1418415112).
  *PNAS*, 112, 10056–10061.
- Rindi, L., Dal Bello, M. & Benedetti-Cecchi, L. (2018). [Experimental
  evidence of spatial signatures of approaching regime shifts in macroalgal
  canopies](https://doi.org/10.1002/ecy.2391). *Ecology*, 99, 1709–1715.
- van Belzen, J. et al. (2017). [Vegetation recovery in tidal marshes reveals
  critical slowing down under increased inundation](https://doi.org/10.1038/ncomms15811).
  *Nature Communications*, 8, 15811.

## Screened but not included in the current run

- Drake & Griffen (2010), Daphnia extinction: the repository download endpoint
  denied the current automated request.
- Veraart et al. (2012), cyanobacterial chemostats: the public data endpoint
  returned an anti-bot HTML document rather than the advertised data file.
- Butitta et al. (2017), whole-lake manipulation: package metadata were public,
  but the data entity was not anonymously retrievable during this audit.

These records remain candidates. Failed or HTML downloads are never parsed as
data and do not enter any reported statistics.
