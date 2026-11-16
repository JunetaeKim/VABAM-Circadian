<!-- Paste into VABAM-official/README.md, e.g. after "Research Highlights". -->

## Spin-off: VABAM-Circadian

[**VABAM-Circadian**](https://github.com/JunetaeKim/VABAM-Circadian) applies VABAM, unchanged, to a different signal class and time scale: ~24-h circadian rhythms of *Bmal1*/*Per2*-luciferase reporters recorded every 10 min in cancer cell lines (Ector et al., *Mol Syst Biol* 2025). Re-deriving the filter cut-off range (`FcLimit`) for 10-min sampling lets θ modulate the amplitude of circadian waveforms while their phase and period are preserved. The spin-off adds circadian preprocessing, multi-source training with external test sets, label-free model selection, and downstream analyses linking the learned representations to clock-gene knockouts, circadian phenotypes, drug response and patient transcriptomes, including a comparison of θ + z with single-z VAE representations.
