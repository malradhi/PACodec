import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import librosa

MAX_WAV_VALUE = 32768.0

# Global cache for f0 values
f0_cache = {}


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output
    
    
def f0_to_one_hot(f0_hz, n_bins=513, sample_rate=22050, n_fft=1024):

    B, T = f0_hz.shape

    freq_bins = torch.linspace(0, sample_rate // 2, steps=n_bins, device=f0_hz.device)  # (N,)
    f0_bin_idx = torch.bucketize(f0_hz, freq_bins)  # (B, T), gives index in range [0, N]
    f0_bin_idx = torch.clamp(f0_bin_idx, 0, n_bins - 1)

    one_hot = torch.zeros(B, n_bins, T, device=f0_hz.device)
    for b in range(B):
        one_hot[b].scatter_(0, f0_bin_idx[b].unsqueeze(0), 1.0)

    return one_hot


def extract_f0(audio, sampling_rate, hop_size, f0_min=80, f0_max=750, frame_length=1024):

    if isinstance(audio, torch.Tensor):
        audio_np = audio.numpy()
    else:
        audio_np = audio
        
    # Extract f0 using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_np, 
        fmin=f0_min,
        fmax=f0_max,
        sr=sampling_rate,
        hop_length=hop_size,
        frame_length=frame_length
    )
    
    # Replace NaN values with zeros (unvoiced)
    f0 = np.nan_to_num(f0)
    
    # Convert to torch tensor
    f0 = torch.FloatTensor(f0)
    
    return f0


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.view_as_real(spec) # (B, N, T, 2)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def complex_components(y, n_fft, hop_size, win_size, center=False):
    global hann_window
    hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    # (B, N, T)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.view_as_real(spec) # (B, N, T, 2)
    mag = torch.sqrt(spec.pow(2).sum(-1)+(1e-9)) # (B, N, T)
    
    # Calculate power spectrum
    power = mag.pow(2) # (B, N, T)
    
    phase = torch.angle(spec.sum(-1)) # (B, N, T)
    spec = spec.permute(0,3,1,2) #(B, 2, N, T)
    
    #* (B, 5, N, T) - now including power spectrum
    complex_comp = torch.cat((mag.unsqueeze(1), phase.unsqueeze(1), power.unsqueeze(1)), dim=1)
    return complex_comp


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class ComplexDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, 
                 split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, data_path=None,
                 f0_min=80, f0_max=750, f0_frame_length=1024, f0_cache_dir="/ria/AutoVocoder/f0_cache/"):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.data_path = data_path
        
        # F0 extraction parameters
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_frame_length = f0_frame_length
        self.f0_cache_dir = f0_cache_dir
        
        # Create f0 cache directory if it doesn't exist
        if self.f0_cache_dir is not None:
            os.makedirs(self.f0_cache_dir, exist_ok=True)

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        # Get f0 with proper caching
        f0 = self.get_f0(filename, audio.squeeze(0).numpy())

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                    
                    # Generate complex components
                    complx = complex_components(audio, self.n_fft, self.hop_size, self.win_size, center=False)
                    
                    # Calculate corresponding f0 segment based on the actual size of complex components
                    frames_in_complx = complx.size(3)  # Get the exact number of frames
                    f0_start = int(audio_start / self.hop_size)
                    
                    # Ensure f0 is exactly the right length to match complx
                    if f0_start + frames_in_complx <= f0.size(0):
                        f0 = f0[f0_start:f0_start+frames_in_complx]
                    else:
                        f0_segment = f0[f0_start:]
                        f0 = torch.nn.functional.pad(f0_segment, (0, frames_in_complx - f0_segment.size(0)), 'constant')
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                    
                    # Generate complex components
                    complx = complex_components(audio, self.n_fft, self.hop_size, self.win_size, center=False)
                    
                    # Get exact number of frames in complx
                    frames_in_complx = complx.size(3)
                    
                    # Ensure f0 is exactly right length
                    if f0.size(0) > frames_in_complx:
                        f0 = f0[:frames_in_complx]
                    else:
                        f0 = torch.nn.functional.pad(f0, (0, frames_in_complx - f0.size(0)), 'constant')
            else:
                # Generate complex components
                complx = complex_components(audio, self.n_fft, self.hop_size, self.win_size, center=False)
                
                # Adjust f0 to match complx frame count
                frames_in_complx = complx.size(3)
                if f0.size(0) > frames_in_complx:
                    f0 = f0[:frames_in_complx]
                else:
                    f0 = torch.nn.functional.pad(f0, (0, frames_in_complx - f0.size(0)), 'constant')
        else:
            complx = np.load(
                os.path.join(self.data_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            complx = torch.from_numpy(complx)

            if len(complx.shape) < 4:
                complx = complx.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, complx.size(3) - frames_per_seg - 1)
                    complx = complx[:, :, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                    
                    # Get corresponding f0 segment based on actual frames in complx
                    frames_in_complx = complx.size(3)
                    if f0.size(0) > mel_start + frames_in_complx:
                        f0 = f0[mel_start:mel_start + frames_in_complx]
                    else:
                        f0_segment = f0[mel_start:]
                        f0 = torch.nn.functional.pad(f0_segment, (0, frames_in_complx - f0_segment.size(0)), 'constant')
                else:
                    original_frames = complx.size(3)
                    complx = torch.nn.functional.pad(complx, (0, frames_per_seg - complx.size(3)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                    
                    # Update frames_in_complx after padding
                    frames_in_complx = complx.size(3)
                    
                    # Ensure f0 matches exactly
                    if f0.size(0) > frames_in_complx:
                        f0 = f0[:frames_in_complx]
                    else:
                        f0 = torch.nn.functional.pad(f0, (0, frames_in_complx - f0.size(0)), 'constant')

        # Reshape f0 to match the batch dimension of complx
        f0 = f0.unsqueeze(0)  # Shape becomes [1, T]
        
        # Verify shapes match in time dimension
        assert complx.size(3) == f0.size(1), f"Shape mismatch: complx has {complx.size(3)} frames but f0 has {f0.size(1)} frames"

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (complx.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), f0)

    def get_f0(self, filename, audio):
        """
        Get the f0 for the audio file, either from cache or by computing it
        
        Args:
            filename (str): Path to audio file
            audio (numpy.ndarray): Audio data as numpy array
            
        Returns:
            torch.Tensor: F0 values for the audio
        """
        # Generate cache filename
        if self.f0_cache_dir is not None:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            f0_cache_path = os.path.join(self.f0_cache_dir, f"{base_name}_f0.npy")
        else:
            # Use in-memory cache
            f0_cache_path = None
            
        # Check if f0 exists in global memory cache
        if filename in f0_cache:
            return f0_cache[filename]
            
        # Check if f0 exists in file cache
        if f0_cache_path is not None and os.path.exists(f0_cache_path):
            try:
                f0 = torch.from_numpy(np.load(f0_cache_path))
                # Store in memory cache too
                f0_cache[filename] = f0
                return f0
            except Exception as e:
                print(f"Error loading cached f0 for {filename}: {e}. Recomputing.")
                
        # If not in cache, compute f0
        f0 = extract_f0(
            audio, 
            self.sampling_rate, 
            self.hop_size, 
            self.f0_min, 
            self.f0_max, 
            self.f0_frame_length
        )
        
        # Store in memory cache
        f0_cache[filename] = f0
        
        # Store in file cache if enabled
        if f0_cache_path is not None:
            try:
                np.save(f0_cache_path, f0.numpy())
            except Exception as e:
                print(f"Error saving f0 to cache for {filename}: {e}")
                
        return f0

    def __len__(self):
        return len(self.audio_files)