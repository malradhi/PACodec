<h1 align="center"><strong>Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding in the Complex Spectrum</strong></h1>

<p align="center" style="font-size: 1 em; margin-top: 1em">
<a href="https://malradhi.github.io/">Mohammed Salah Al-Radhi</a>, 
Riad Larbi,
<a href="https://www.semanticscholar.org/author/M%C3%A1ty%C3%A1s-Bartalis/3194027">Mátyás Bartalis</a>,
<a href="https://scholar.google.ro/citations?user=Qf5PHwoAAAAJ&hl=en/">Géza Németh</a>,
</p>

<p align="center">
  Department of Telecommunications and Artificial Intelligence, Budapest University of Technology and Economics, Budapest, Hungary<br>
  
</p>

<div align="center">
<!--   <a href="https://github.com/ZhikangNiu/A-DMA">
    <img src="https://img.shields.io/badge/Python-3.10-brightgreen" alt="Python">
  </a> -->
<!--   <a href="https://arxiv.org/abs/2505.19595v1">
    <img src="https://img.shields.io/badge/arXiv-2505.19595-b31b1b.svg?logo=arXiv" alt="arXiv">
  </a> -->
  <a href="https://malradhi.github.io/PACodec/">
    <img src="https://img.shields.io/badge/GitHub-Demo%20page-orange.svg" alt="Demo" width="180">
  </a>
</div

 
 
<br>
<br> 


## 📚 Table of Contents
- [📜 News](#-news)
- [💡 Highlights](#-highlights)
- [🛠️ Usage](#️-usage)
- [📂 Directory Structure](#-directory-structure)
- [🙏 Acknowledgements](#-acknowledgements)
- [📖 Citation](#-citation)



<br>
<br>


## 📜 News
🧠 [2025.09.18] We’ve officially submitted our paper to ICASSP 2026! 🎉

🧠 [2025.09.18] We’ve also released the full source code for our work — check the src folder for details.

<br>
<br> 

## 💡 Highlights
1. **Multi-Modal iEEG Feature Encoding**: MiSTR introduces a wavelet-based encoder combined with prosody-aware features (pitch, energy, shimmer, duration, phase variability) to model the neural dynamics of speech production.
2. **Transformer-Based Prosody Decoder**: A novel Transformer architecture captures long-range dependencies in brain activity to predict expressive and fluent Mel spectrograms aligned with speech prosody.
3. **Neural Phase Vocoder (IHPR)**: MiSTR proposes Iterative Harmonic Phase Reconstruction (IHPR), ensuring phase continuity and harmonic consistency for high-fidelity audio synthesis without vocoder artifacts.
4. **State-of-the-Art Performance**: Achieves a Pearson correlation of 0.91, STOI of 0.73, and MOSA score of 3.38, outperforming all existing baselines in iEEG-to-speech synthesis.
5. **Clinically Inspired Design**: Designed with speech neuroprosthetics in mind, MiSTR offers a scalable, robust pipeline for restoring natural speech in individuals with severe communication impairments.
6. **Code and Samples Available**: Full implementation, pretrained models, and inference samples are provided in this GitHub repository to support reproducibility and further research.



<br>
<br>

## 🛠️ Usage

### 1. Clone the Repository and Set Up the Environment

```bash
git clone git clone https://github.com/malradhi/mistr.git
cd mistr

# Recommended: Create a clean environment
conda create -n mistr python=3.10
conda activate mistr

# Install required Python packages
pip install -r requirements.txt

```
💡 Note: Requires PyTorch ≥ 2.0 and CUDA-compatible GPU for best performance.


<br>

### 2. Feature Extraction from iEEG and Audio

```bash
python neural_signal_encoder.py

```

This will generate the following files for each participant (e.g., `sub-XX`):

- `*_feat.npy`: Wavelet + prosody features extracted from iEEG
- `*_spec.npy`: Ground-truth Mel spectrogram from original audio
- `*_prosody.npy`: Extracted prosody features (pitch, energy, shimmer, duration, phase variability)

<br>

### 3. Train the Spectrogram Mapper (Autoencoder + Transformer)

```bash
python spectrogram_mapper_transformer.py

```

This script will:

- Train a **neural autoencoder** to compress iEEG features into a compact latent space
- Use a **Transformer** to predict Mel spectrograms from the latent iEEG representations
- Generate audio waveforms from predicted and ground-truth spectrograms
- Save predicted spectrograms as `*_predicted_spec.npy`
- Save synthesized audio files:
  - `*_orig_synthesized.wav` — from original spectrogram
  - `*_predicted.wav` — from predicted spectrogram
- Save evaluation results in `temporal_attention_results.npy`


<br>

### 4. Phase-Aware Waveform Reconstruction (IHPR Vocoder)

```bash
python harmonic_phase_reconstructor.py

```

This script applies **Iterative Harmonic Phase Reconstruction (IHPR)** to refine phase and improve audio quality.

It will:

- Load predicted spectrograms (`*_predicted_spec.npy`)
- Apply harmonic-consistent phase reconstruction
- Save high-fidelity audio waveforms for each participant:
  - `*_predicted.wav` (updated)
  - `*_orig_synthesized.wav` (if regenerated)
- Output `.wav` files in the `/results/` directory


<br>

### 5. Visualization of Results

```bash
python neural_output_visualizer.py

```

This script generates high-resolution plots and visualizations:

- `results.png`: Participant-wise correlation scores
- `spec_example.png` and `.pdf`: Ground-truth vs. predicted spectrograms
- `wav_example.png` and `.pdf`: Waveform comparison of original vs. reconstructed audio
- `*_prosody_visualization.png`: Plots of extracted prosody features (if available)

All visual outputs are saved in the `/results/` directory.



<br>
<br> 

## 📂 Directory Structure

```bash
./features/            # Extracted features and prosody files
./results/             # Output spectrograms, waveforms, and plots
./harmonic_phase_reconstructor.py
./neural_signal_encoder.py
./spectrogram_mapper_transformer.py
./neural_output_visualizer.py
```


<br>
<br>

## 🙏 Acknowledgements

This work is supported by the **European Union’s HORIZON Research and Innovation Programme** under grant agreement No. **101120657**, project **[ENFIELD](https://doi.org/10.3030/101120657)** (European Lighthouse to Manifest Trustworthy and Green AI), and by the Ministry of Innovation and Culture and the National Research, Development and Innovation Office of Hungary within the framework of the National Laboratory of Artificial Intelligence.

**M.S. Al-Radhi’s** research was supported by the project **EKÖP-24-4-II-BME-197**, through the National Research, Development and Innovation (NKFI) Fund.



<br>
<br>




## 📖 Citation and License

We’ve released our code under the MIT License to support open research. If you use it in your work, please consider citing us:


```bibtex
@inproceedings{alradhi2025mistr,
  title     = {MiSTR: Multi-Modal iEEG-to-Speech Synthesis with Transformer-Based Prosody Prediction and Neural Phase Reconstruction},
  author    = {Mohammed Salah Al-Radhi and G{\'e}za N{\'e}meth and Branislav Gerazov},
  booktitle = {Proceedings of Interspeech 2025},
  year      = {2025},
  address   = {Rotterdam, The Netherlands}
}
```




