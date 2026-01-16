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

    def _preprocess_audio(self, audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Resample and normalize audio."""
        target_sr = self.sample_rate
        
        if sampling_rate != target_sr:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
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
