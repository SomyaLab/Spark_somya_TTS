import sys
from pathlib import Path

# Add Spark-TTS to path (<repo>/Spark-TTS).
# This file lives at: <repo>/src/spark_tts/data/tokenizer.py
_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "Spark-TTS"))

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
        # Limit peak VRAM during wav2vec2/codec tokenize by chunking batch dimension.
        # This is intentionally conservative; increase if you have plenty of VRAM.
        self.max_tokenize_chunk: int = 8

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
        # Autocast improves throughput on Ampere+ for wav2vec2 forward.
        if str(self.tokenizer.feature_extractor.device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_output = self.tokenizer.feature_extractor(input_values)
        else:
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
    def tokenize_audio_ids(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
    ) -> tuple[list[int], list[int]]:
        """
        Convert audio array to (global_ids, semantic_ids).

        This avoids the intermediate "<|bicodec_*_N|>" string format so callers can
        build model `input_ids` directly without re-tokenizing text.
        """
        # Use the same all-GPU path as the batch method to avoid CPU round-trips.
        audio_array = self._preprocess_audio(audio_array, sampling_rate)
        ref_wav_np = self.tokenizer.get_ref_clip(audio_array)

        device = torch.device(self.device)
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(device)
        ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(device)

        processed = self.tokenizer.processor(
            [audio_array],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values.to(self.tokenizer.feature_extractor.device)

        if str(self.tokenizer.feature_extractor.device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_output = self.tokenizer.feature_extractor(input_values)
        else:
            model_output = self.tokenizer.feature_extractor(input_values)

        if model_output.hidden_states is None:
            raise ValueError(
                "Wav2Vec2Model did not return hidden states. "
                "Ensure config `output_hidden_states=True`."
            )

        feats_mix = (
            model_output.hidden_states[11]
            + model_output.hidden_states[14]
            + model_output.hidden_states[16]
        ) / 3

        batch = {"wav": audio_tensor, "ref_wav": ref_wav_tensor, "feat": feats_mix.to(device)}

        if str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                semantic_token_ids, global_token_ids = self.tokenizer.model.tokenize(batch)
        else:
            semantic_token_ids, global_token_ids = self.tokenizer.model.tokenize(batch)
        global_ids = global_token_ids.squeeze().detach().cpu().tolist()
        semantic_ids = semantic_token_ids.squeeze().detach().cpu().tolist()
        return global_ids, semantic_ids

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
        global_ids, semantic_ids = self.tokenize_audio_ids(audio_array, sampling_rate)
        global_tokens = "".join(f"<|bicodec_global_{i}|>" for i in global_ids)
        semantic_tokens = "".join(f"<|bicodec_semantic_{i}|>" for i in semantic_ids)
        return global_tokens, semantic_tokens

    @torch.inference_mode()
    def tokenize_audio_batch_ids(
        self,
        audio_arrays: list[np.ndarray],
        sampling_rates: list[int],
    ) -> list[tuple[list[int], list[int]]]:
        """
        Convert a batch of audio arrays to (global_ids, semantic_ids) per sample.
        """
        if not audio_arrays:
            return []

        # Preprocess all audio (resample, normalize)
        processed_audios: list[np.ndarray] = []
        ref_wavs: list[np.ndarray] = []
        for audio, sr in zip(audio_arrays, sampling_rates):
            processed = self._preprocess_audio(audio, sr)
            ref_wav = self.tokenizer.get_ref_clip(processed)
            processed_audios.append(processed)
            ref_wavs.append(ref_wav)

        # Pad audio to same length for batching
        max_audio_len = max(len(a) for a in processed_audios)
        max_ref_len = max(len(r) for r in ref_wavs)

        padded_audios = []
        for audio in processed_audios:
            if len(audio) < max_audio_len:
                audio = np.pad(audio, (0, max_audio_len - len(audio)), mode="constant")
            padded_audios.append(audio)

        padded_refs = []
        for ref in ref_wavs:
            if len(ref) < max_ref_len:
                ref = np.pad(ref, (0, max_ref_len - len(ref)), mode="constant")
            padded_refs.append(ref)

        audio_tensor = torch.from_numpy(np.stack(padded_audios)).float().to(self.device)
        ref_tensor = torch.from_numpy(np.stack(padded_refs)).float().to(self.device)

        processed = self.tokenizer.processor(
            # Use unpadded audio for wav2vec2; processor handles padding efficiently.
            [a for a in processed_audios],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values_all = processed.input_values.to(self.tokenizer.feature_extractor.device)

        # Chunk wav2vec2 forward to reduce peak memory (hidden_states are heavy).
        bsz = int(input_values_all.shape[0])
        chunk = max(1, int(getattr(self, "max_tokenize_chunk", 8)))
        feats_chunks: list[torch.Tensor] = []
        for start in range(0, bsz, chunk):
            end = min(bsz, start + chunk)
            input_values = input_values_all[start:end]
            if str(self.tokenizer.feature_extractor.device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model_output = self.tokenizer.feature_extractor(input_values)
            else:
                model_output = self.tokenizer.feature_extractor(input_values)

            if model_output.hidden_states is None:
                raise ValueError(
                    "Wav2Vec2Model did not return hidden states. "
                    "Ensure config `output_hidden_states=True`."
                )

            feats_mix = (
                model_output.hidden_states[11]
                + model_output.hidden_states[14]
                + model_output.hidden_states[16]
            ) / 3
            feats_chunks.append(feats_mix.to(self.device))

        feats_all = torch.cat(feats_chunks, dim=0)

        # Chunk codec tokenize as well to avoid OOM on long audio.
        results: list[tuple[list[int], list[int]]] = []
        for start in range(0, bsz, chunk):
            end = min(bsz, start + chunk)
            batch = {
                "wav": audio_tensor[start:end],
                "ref_wav": ref_tensor[start:end],
                "feat": feats_all[start:end],
            }

            if str(self.device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    semantic_token_ids, global_token_ids = self.tokenizer.model.tokenize(batch)
            else:
                semantic_token_ids, global_token_ids = self.tokenizer.model.tokenize(batch)

            semantic_ids = semantic_token_ids.detach().cpu().tolist()
            global_ids = global_token_ids.detach().cpu().tolist()

            for g, s in zip(global_ids, semantic_ids):
                g_ids = [int(x) for x in (g[0] if isinstance(g, list) and g and isinstance(g[0], list) else g)]
                s_ids = [int(x) for x in (s[0] if isinstance(s, list) and s and isinstance(s[0], list) else s)]
                results.append((g_ids, s_ids))

        return results

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
        id_results = self.tokenize_audio_batch_ids(audio_arrays, sampling_rates)
        results: list[tuple[str, str]] = []
        for global_ids, semantic_ids in id_results:
            global_tokens = "".join(f"<|bicodec_global_{idx}|>" for idx in global_ids)
            semantic_tokens = "".join(f"<|bicodec_semantic_{idx}|>" for idx in semantic_ids)
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
