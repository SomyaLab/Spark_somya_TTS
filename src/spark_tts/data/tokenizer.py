import sys
from pathlib import Path

# Add Spark-TTS to path (relative to project root)
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root / "Spark-TTS"))

import torch
import numpy as np
import torchaudio.transforms as T

from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize


class AudioTokenizer:
    """Wrapper around BiCodecTokenizer for audio tokenization."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = BiCodecTokenizer(model_dir, device)
        self.config = self.tokenizer.config
        self.sample_rate = self.config.get("sample_rate", 16000)
        # Cache resamplers by source sample rate
        self._resamplers: dict[int, T.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        """Get or create cached resampler for given sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
        return self._resamplers[orig_sr]

    def _preprocess_audio(self, audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Resample and normalize audio."""
        target_sr = self.sample_rate
        
        if sampling_rate != target_sr:
            resampler = self._get_resampler(sampling_rate)
            audio_tensor = torch.from_numpy(audio_array).float()
            audio_array = resampler(audio_tensor).numpy()
        
        if self.config.get("volume_normalize", False):
            audio_array = audio_volume_normalize(audio_array)
        
        return audio_array

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """Extract wav2vec2 features from single audio tensor."""
        wav_np = wavs.squeeze(0).cpu().numpy()
        processed = self.tokenizer.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values.to(self.tokenizer.feature_extractor.device)
        model_output = self.tokenizer.feature_extractor(input_values)

        if model_output.hidden_states is None:
            raise ValueError(
                "Wav2Vec2Model did not return hidden states. "
                "Ensure config `output_hidden_states=True`."
            )

        num_layers = len(model_output.hidden_states)
        required_layers = [11, 14, 16]
        if any(layer >= num_layers for layer in required_layers):
            raise IndexError(
                f"Requested hidden state indices {required_layers} "
                f"out of range for model with {num_layers} layers."
            )

        feats_mix = (
            model_output.hidden_states[11]
            + model_output.hidden_states[14]
            + model_output.hidden_states[16]
        ) / 3

        return feats_mix

    @torch.inference_mode()
    def tokenize_audio(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
    ) -> tuple[str, str]:
        """
        Convert audio array to global and semantic tokens.

        Args:
            audio_array: Audio samples as numpy array
            sampling_rate: Sample rate of input audio

        Returns:
            Tuple of (global_tokens_str, semantic_tokens_str)
        """
        audio_array = self._preprocess_audio(audio_array, sampling_rate)
        ref_wav_np = self.tokenizer.get_ref_clip(audio_array)

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(self.device)
        ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(self.device)

        feat = self.extract_wav2vec2_features(audio_tensor)

        batch = {
            "wav": audio_tensor,
            "ref_wav": ref_wav_tensor,
            "feat": feat.to(self.device),
        }

        semantic_token_ids, global_token_ids = self.tokenizer.model.tokenize(batch)

        global_tokens = "".join(
            f"<|bicodec_global_{i}|>"
            for i in global_token_ids.squeeze().cpu().numpy()
        )
        semantic_tokens = "".join(
            f"<|bicodec_semantic_{i}|>"
            for i in semantic_token_ids.squeeze().cpu().numpy()
        )

        return global_tokens, semantic_tokens

    @torch.inference_mode()
    def tokenize_audio_batch(
        self,
        audio_arrays: list[np.ndarray],
        sampling_rates: list[int],
    ) -> list[tuple[str, str]]:
        """
        Convert a batch of audio arrays to global and semantic tokens.
        
        Args:
            audio_arrays: List of audio samples as numpy arrays
            sampling_rates: List of sample rates for each audio
            
        Returns:
            List of tuples (global_tokens_str, semantic_tokens_str)
        """
        if not audio_arrays:
            return []
        
        # Preprocess all audio (resample, normalize)
        processed_audios = []
        ref_wavs = []
        for audio, sr in zip(audio_arrays, sampling_rates):
            processed = self._preprocess_audio(audio, sr)
            ref_wav = self.tokenizer.get_ref_clip(processed)
            processed_audios.append(processed)
            ref_wavs.append(ref_wav)
        
        # Pad audio to same length for batching
        max_audio_len = max(len(a) for a in processed_audios)
        max_ref_len = max(len(r) for r in ref_wavs)
        
        # Pad and stack audio tensors
        padded_audios = []
        for audio in processed_audios:
            if len(audio) < max_audio_len:
                audio = np.pad(audio, (0, max_audio_len - len(audio)), mode='constant')
            padded_audios.append(audio)
        
        padded_refs = []
        for ref in ref_wavs:
            if len(ref) < max_ref_len:
                ref = np.pad(ref, (0, max_ref_len - len(ref)), mode='constant')
            padded_refs.append(ref)
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(np.stack(padded_audios)).float().to(self.device)
        ref_tensor = torch.from_numpy(np.stack(padded_refs)).float().to(self.device)
        
        # Extract wav2vec2 features for the batch
        # The processor handles batched numpy arrays
        wav_list = [a for a in padded_audios]  # List of numpy arrays for processor
        processed = self.tokenizer.processor(
            wav_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values.to(self.device)
        model_output = self.tokenizer.feature_extractor(input_values)
        
        feats_mix = (
            model_output.hidden_states[11]
            + model_output.hidden_states[14]
            + model_output.hidden_states[16]
        ) / 3
        
        # BiCodec tokenization
        batch = {
            "wav": audio_tensor,
            "ref_wav": ref_tensor,
            "feat": feats_mix.to(self.device),
        }
        
        semantic_token_ids, global_token_ids = self.tokenizer.model.tokenize(batch)
        
        # Convert to token strings for each sample in batch
        results = []
        batch_size = len(audio_arrays)
        semantic_ids = semantic_token_ids.cpu().numpy()
        global_ids = global_token_ids.cpu().numpy()
        
        for i in range(batch_size):
            global_tokens = "".join(
                f"<|bicodec_global_{idx}|>"
                for idx in global_ids[i]
            )
            semantic_tokens = "".join(
                f"<|bicodec_semantic_{idx}|>"
                for idx in semantic_ids[i]
            )
            results.append((global_tokens, semantic_tokens))
        
        return results
    
    @staticmethod
    def load_audio_fast(audio_path: str) -> tuple[np.ndarray, int]:
        """Load audio using soundfile (fast and reliable for WAV)."""
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    def detokenize(
        self,
        global_ids: torch.Tensor,
        semantic_ids: torch.Tensor,
    ) -> np.ndarray:
        """Convert token IDs back to audio waveform."""
        self.tokenizer.model.to(self.device)
        return self.tokenizer.detokenize(
            global_ids.to(self.device),
            semantic_ids.to(self.device),
        )

    def offload_to_cpu(self):
        """Move models to CPU to free GPU memory."""
        self.tokenizer.model.cpu()
        self.tokenizer.feature_extractor.cpu()
        torch.cuda.empty_cache()

    def to(self, device: str):
        """Move models to specified device."""
        self.device = device
        self.tokenizer.model.to(device)
        self.tokenizer.feature_extractor.to(device)
