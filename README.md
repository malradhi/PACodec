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
1. **Prosody-Guided Harmonic Attention**: Introduces an F0-driven attention mechanism that emphasises voiced regions and harmonic structures, reducing pitch drift and improving prosodic fidelity.
2. **Direct Complex-Spectrum Prediction**: Unlike mel-spectrogram–based vocoders, the model directly predicts real and imaginary STFT components, ensuring phase-coherent waveform reconstruction.
3. **Multi-Objective Perceptual Training**: Combines MR-STFT, adversarial, and novel phase-aware losses to jointly optimise spectral fidelity, phase continuity, and perceptual quality.
4. **Robust Performance Gains**: Achieves a 22% reduction in F0 RMSE, 18% lower voiced/unvoiced error, and MOS improvement of +0.15 over HiFi-GAN and AutoVocoder baselines.
5. **Natural and Expressive Speech**: Preserves sharper harmonics, temporal coherence, and pitch accuracy, resulting in more natural and robust synthetic speech.
6. **Open Resources**: Full source code and demo samples are available in the GitHub repository to support reproducibility and future research.





<br>
<br>

## 🛠️ Usage

### 1. Clone the Repository and Set Up the Environment

```bash
git clone git clone https://github.com/malradhi/pacodec.git
cd pacodec

# Recommended: Create a clean environment
conda create -n pacodec python=3.10
conda activate pacodec

# Install required Python packages
pip install -r requirements.txt

```
💡 Note: Requires PyTorch ≥ 2.0 and CUDA-compatible GPU for best performance.


<br>

### 2. Prepare Dataset

PACodec expects a dataset of .wav files and training/validation split text files (training.txt, validation.txt).
Each line in the text file should list the audio file basename and optional metadata.

Update --input_wavs_dir, --input_training_file, and --input_validation_file in your config.

<br>


### 3. Feature Extraction (Complex Spectrum + F0)

The dataset loader (ComplexDataset) automatically extracts:
- Complex spectral components (magnitude, phase, power)
- Fundamental frequency (F0) using librosa.pyin with caching
- Ground-truth Mel spectrogram for loss computation

```bash
python complexdataset.py

```

All features are aligned and cached for efficient training.

<br>


### 4. Train PACodec

```bash
python train.py --config config.json --checkpoint_path cp_pacodec

```

This script will:
- Initialize the Encoder (with F0-guided masking + residual attention)
- Train the Generator to predict complex STFT components
- Optimize with adversarial discriminators (Multi-Period & Multi-Scale)
- Use combined mel loss, waveform loss, and feature matching loss
- Save checkpoints (e_xxxxxx, g_xxxxxx, do_xxxxxx) and TensorBoard logs in cp_pacodec/


<br>

### 5. Inference (Waveform Reconstruction)

Once trained, the Generator can directly reconstruct audio:
- Loads encoded latent features
- Predicts magnitude & phase
- Uses inverse STFT (TorchSTFT.polar_inverse) for waveform synthesis
Resulting .wav files are saved in the /results/ directory.


<br>

### 6. Visualisation

```bash
cp_pacodec/logs/
```

You can monitor them using TensorBoard:

```bash
tensorboard --logdir cp_pacodec/logs
```



<br>
<br> 

## 📂 Directory Structure

```bash
./complexdataset.py        # Data pipeline: audio, F0, complex spectrum
./models.py                # Encoder, Generator, Discriminators
./train.py                 # Training loop with losses & checkpointing
./config.json              # Model and training hyperparameters
./cp_pacodec/              # Saved checkpoints and logs
./results/                 # Generated waveforms and spectrograms
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
  title     = {Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding in the Complex Spectrum},
  author    = {Mohammed Salah Al-Radhi and G{\'e}za N{\'e}meth and Branislav Gerazov},
  booktitle = {Proceedings of ICASSP},
  year      = {2026},
  address   = {Barcelona, Spain}
}
```




