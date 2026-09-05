# QIR source and candidate register

Checked 2026-09-05. Retrieval confirms the resources below; it does not certify a
candidate as a ready-to-run independent recovery dataset.

## Existing real-data baseline

- van Belzen et al. 2017, DOI https://doi.org/10.1038/ncomms15811.
- Original bundle: https://zenodo.org/records/4998258, 60,639,751-byte ZIP,
  MD5 `eb545ba9e8e68e41c7f2988157e447c7`. Locally downloaded and checksum-verified.
- Includes two sites, binary images, static study masks, inundation maps and original
  MATLAB/R scripts. Read as data; third-party scripts are not executed by the adapter.
- Limitations: few acquisitions, two related sites, retrospective environmental map.

## Irregular satellite representations

- ALISE official implementation: https://github.com/irisdum/alise.
- Paper: https://arxiv.org/abs/2407.08448.
- Repository declares AGPL-3.0 and provides an ONNX inference route; inputs described
  in its README use 10 Sentinel-2 bands and irregular sequences. The current marsh
  binary imagery does not match those pretrained spectral channels.
- Decision: reference architecture only in phase one; no copied code or pretrained
  model dependency. Reassess when an appropriate multispectral second dataset exists.
- Core neural implementation uses PyTorch GRU/Conv2d; docs:
  https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html.
  Reproducibility limits: https://docs.pytorch.org/docs/2.9/notes/randomness.html.

## Wenchuan independent-system candidate

- Public multi-temporal inventory: https://zenodo.org/records/1484667,
  DOI https://doi.org/10.5281/zenodo.1484667.
- The record describes georeferenced polygons over 471 square kilometres for
  2005-2018 plus a separate debris-flow/rainfall dataset. File list includes
  `Dataset1_Mapping.rar`, `Dataset2_DF_TrigRain.rar` and Readme documents.
- This is an available **disturbance inventory**, not yet a complete time-varying
  vegetation recovery raster benchmark. Needs image acquisition dates, image-level QA,
  pre-disturbance reference, environment availability, recovery definitions and
  geographically independent holdouts before admission.
- Adjacent prior work: https://doi.org/10.1016/j.indic.2026.101300,
  "Probabilistic estimation of vegetation greenness recovery timeline in a
  landslide-prone region affected by the 2008 Wenchuan earthquake".
  Search metadata was found; full article retrieval failed during this session.
  Do not assert a research gap until methods, data and validation are fully compared.

## Admission criteria

Require downloadable and licensed temporal rasters (or reproducible acquisition),
dated disturbance labels, persistent recovery/end-of-follow-up definitions, explicit
unrecovered units, image masks/quality, prediction-time environmental covariates,
and an independent spatial split. Record hashes and preprocessing commands.
Candidates that supply only aggregate tables or single-date segmentation labels do
not satisfy the second-system milestone. No second-system experiment has run yet.

