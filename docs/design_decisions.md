# Design decisions

## 0. Relation to the official code

- The official VABAM repository is imported, never modified or copied. Paths the official code writes to by itself (`__pycache__`, `./Logs/`, `./Results/`) are redirected as described in the header of `src/vabam_bridge.py`; the full pipeline was run against a read-only copy of the official code to confirm that it attempts no writes.
- Environment: the official code was validated on TF 2.4/2.10. The pilot also ran on TF 2.20 with `tf-keras` (Keras 2 API, required for `add_loss` / `add_metric`; `TF_USE_LEGACY_KERAS=1` is set by the bridge).

## 1. Data

- Configuration A uses the MSB 2025 raw file only: all 180 recordings (15 breast cell models, MCF10A, U-2 OS lines). Labels are never used for training. `preprocess.split.exclude_labels` holds U-2 OS out entirely when needed.

## 2. Preprocessing

| Step | Choice | Reason |
|---|---|---|
| Detrending | Blackman windowed-sinc low-pass, T_c = 48 h; detrended = raw − trend | Same method and setting as the authors' pyBOAT pipeline (`sinc_smooth`). All 180 traces agree with the authors' output (r = 1.0000, median max relative error 0.00016) |
| Normalisation | **relative**: detrended / trend | Removes reporter brightness and its slow drift while keeping relative amplitude and damping, which θ is meant to control. Envelope normalisation (amplitude removed) is optional only |
| Window length | 400 samples = 66.7 h | 2.8 cycles at 24 h (2.2–3.3 cycles for 20–30 h). VABAM requires SigDim to be a multiple of 50 and CompSize a multiple of 100 dividing SigDim/50; among lengths covering 2–3 cycles at 10 min only 400 (CompSize 200) is valid |
| Stride | 18 samples = 3 h | 24 windows per complete trace |
| Scaling | x01 = 0.5 + x / (2q), q = 99.5th percentile of \|x\| over training windows (0.406), clipped (0.5 %) | The reconstructor ends in a sigmoid. One global scale keeps relative amplitudes comparable across traces, as the official global min–max does |
| Split | by recording, 15 %, stratified by label | Overlapping windows of one recording never fall on both sides |

## 3. Cut-off range (FcLimit)

- Cut-offs are in cycles/sample and θ ∈ (0, 1) maps to (0, FcLimit); the lower end is fixed at 0 in the official Encoder, so FcLimit is the only adjustable range.
- At 10-min sampling a 24-h rhythm is at 1/144 = 0.00694 cycles/sample. With the official FcLimit = 1.0 every cut-off is far above the circadian band and the low-pass gain at 24 h is 1.00 for every θ: θ controls nothing (`results/filter_response/filter_response_table.txt`).
- **Choice: FcLimit = (10/60)/8 = 0.0208333** (cut-off period from ∞ down to 8 h). Response of the official filters, Depth 1 (kernel 201 samples = 33.5 h):

| θ | Cut-off period | LPF gain at 24 h | LPF gain at 12 h |
|---|---|---|---|
| 0.05 | 160 h | 0.17 | 0.00 |
| 0.25 | 32 h | 0.37 | 0.00 |
| 0.50 | 16 h | 0.83 | 0.17 |
| 0.75 | 10.7 h | 1.00 | 0.68 |
| 0.95 | 8.4 h | 1.00 | 0.96 |

  One θ range sweeps the circadian fundamental and its 12-h harmonic (waveform asymmetry).
- Alternative FcLimit = (10/60)/12 = 0.0138889 (θ = 0.5 ↔ 24-h cut-off): model `SKZFC_CIRC_1_30_200_FC12`.
- Limitation: the kernel (33.5 h) is comparable to the period, so frequency resolution is coarse and 17 % of the 24-h component remains at θ → 0. Depth 2 (kernel 101) never brings the 24-h gain below 0.67. Longer windows improve resolution at the cost of more than three cycles per window (`SKZFC_CIRC_1_30_200_L500_RCB`).

## 4. Training settings changed from the official values

| Item | Official | Here | Reason |
|---|---|---|---|
| FcLimit | 1.0 | 0.0208333 | Section 3 |
| CompSize | 500/800 | 200 | only valid value for SigDim 400 |
| BatSize | 3000 | 256 | 3,598 training windows |
| NEpochs | 5000 | 1500 + early stopping (patience 150) | data size |

All other values (LatDim 30, Depth 1, SpecLosses SKZ+FC, capacities, loss-weight ranges, ReparaStd) equal the official ConfigART800 / ConfigII800.

## 5. Configuration B — multi-source training and external tests

Target: about 3 × 10⁵ training windows, 10 % validation, ≥ 10⁴ external windows, with one complete experiment file held out.

| Decision | Content | Reason |
|---|---|---|
| Nat Commun 2024 | 135 duplicates dropped; 29 new recordings used as external set E2 only | Identical recordings counted twice would inflate the sample size; the new neuroblastoma lines test generalisation |
| Circa-SCOPE time axis | 1 h → 10 min by PCHIP; training stride 1 h | Interpolation adds no information; the filter band (periods ≥ 8 h) is represented at 1 h; finer strides only shift interpolated values |
| Drug/serum addition | No window spans the addition at 108 h; segments 0–108 h and 120 h–end | Keeps acute phase shifts and transients out of windows |
| Split unit | cell (both segments together) for Circa-SCOPE, recording for MSB; stratified by source, label and condition | No overlapping windows of one unit on both sides |
| Scale | q per measurement type (population bioluminescence / single-cell fluorescence) | Relative amplitudes of luminescence and fluorescence are not comparable |
| QC | exclude records whose trend minimum < 0.1 × trend median (26 records) | Prevents the relative normalisation from diverging; fixed before results were seen |
| Balance | fixed source ratio per epoch (MULTI 50:50) | Single-cell windows outnumber population windows about 13:1 |
| Comparison | POP (population only) vs MULTI (+ single cells) on the same validation and external sets, label-free metrics | Whether single-cell data help is decided experimentally |
| HES1 | external set E3 only | Most tracks span one cell cycle (~27 h); only 26 fit a 66.7-h window |

Independent units are reported as recordings and experiment dates (population data) and cells (single-cell data; number of independent experiments unknown for Circa-SCOPE).

## 6. Model selection

Selection uses label-free metrics on the MSB validation recordings excluding U-2 OS: reconstruction R², amplitude ratio between θ_L = 0.95 and 0.05 at fixed z, monotonicity of amplitude in θ_L, and shape preservation. Genotype results and the external sets are never used to choose a configuration.

The inferred θ (encoder mean of the cut-offs) is pulled to its prior and θ is sampled during training, so it is almost constant across inputs. θ is therefore evaluated as a generation-time control, and input-level analyses use z or the model's response to θ.
