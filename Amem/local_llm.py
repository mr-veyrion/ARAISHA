from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional, Tuple

import torch

from .local_config import LocalLLMConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defer llama-cpp import to runtime to avoid conflicts with Intel's ggml
LLAMA_CPP_AVAILABLE = None
Llama = None

# Intel backend import is attempted at runtime within the loader to avoid
# false negatives due to environment-specific import errors during module import.


class LocalLLM:
    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig()
        self.device = self._resolve_device(self.config.device)
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        
        # Backend resolution
        gguf_files = glob.glob(os.path.join(self.config.model_path, "*.gguf"))
        # Decide GGUF based on files; import llama-cpp lazily in loader
        self.is_gguf = len(gguf_files) > 0

        selected_backend = self.config.backend
        if self.is_gguf and selected_backend in ("auto", "gguf"):
            # Prefer GGUF only in auto or explicit gguf mode
            print("[loader] GGUF model detected → using llama.cpp loader")
            self._load_gguf_model(gguf_files[0])
            self.backend = "gguf"
            return
        if selected_backend == "gguf":
            # Explicit GGUF requested but not available
            raise RuntimeError("GGUF backend requested but no .gguf model found or llama-cpp not available")

        # No GGUF present → use standard transformers backend
        print(f"[loader] Using transformers backend on {self.device}")
        self._load_transformers_model()
        self.backend = "transformers"

    def _load_gguf_model(self, gguf_path: str):
        """Load GGUF model using llama-cpp-python."""
        # Import llama-cpp lazily to avoid conflicts
        global Llama, LLAMA_CPP_AVAILABLE
        if Llama is None:
            try:
                from llama_cpp import Llama as _Llama
                Llama = _Llama
                LLAMA_CPP_AVAILABLE = True
            except ImportError:
                LLAMA_CPP_AVAILABLE = False
                raise RuntimeError("llama-cpp-python not installed; GGUF backend unavailable")

        # Load GGUF model with resource-aware settings
        import psutil
        
        # Get system resources
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = os.cpu_count() or 4
        except Exception:
            available_memory_gb = 4.0  # Fallback
            cpu_count = 4
        
        # Adaptive settings based on available resources
        if available_memory_gb < 4:
            # Low memory mode
            n_ctx = 2048
            n_batch = 128
            threads = min(cpu_count // 2, 4)
        elif available_memory_gb < 8:
            # Medium memory mode
            n_ctx = 3072
            n_batch = 256
            threads = min(cpu_count // 2, 6)
        else:
            # High memory mode
            n_ctx = 4096
            n_batch = 512
            threads = min(cpu_count // 2, 8)
        
        print(f"[gguf] Using adaptive settings: {available_memory_gb:.1f}GB RAM → ctx={n_ctx}, batch={n_batch}, threads={threads}")
        
        # Start with resource-appropriate settings
        model_kwargs = {
            "model_path": gguf_path,
            "n_ctx": n_ctx,
            "n_threads": threads,
            "n_batch": n_batch,
            "f16_kv": True,
            "use_mmap": True,
            "verbose": False,
            "use_mlock": False,  # Disable for faster startup
        }
        
        # GPU acceleration if available
        if self.device.startswith("cuda"):
            model_kwargs["n_gpu_layers"] = -1
            
        try:
            self.model = Llama(**model_kwargs)
        except Exception as e:
            # Fallback with minimal settings
            model_kwargs.update({"n_ctx": 2048, "n_batch": 256})
            try:
                self.model = Llama(**model_kwargs)
            except Exception as e2:
                raise RuntimeError(f"GGUF load failed: {e2}")
                
        # Define stop tokens for common models
        self._gguf_stop_tokens = ["</s>", "<|end|>", "<|endoftext|>", "<|eot_id|>"]

    def _load_transformers_model(self):
        """Load model using standard HuggingFace transformers with CPU/GPU support."""
        # Determine dtype and device_map based on hardware
        if self.device.startswith("cuda") and torch.cuda.is_available():
            # GPU: use float16 for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=True
            )
        else:
            # CPU: use float32 for compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                use_cache=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, 
            trust_remote_code=True
        )

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def _infer_dtype(self):
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        if getattr(self, "backend", None) == "gguf":
            # GGUF generation using llama-cpp-python
            return self._generate_gguf(
                prompt, 
                max_new_tokens or self.config.max_new_tokens,
                temperature if temperature is not None else self.config.temperature,
                top_p if top_p is not None else self.config.top_p
            )
        else:
            # Transformers backend
            with torch.inference_mode():
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                    temperature=self.config.temperature if temperature is None else temperature,
                    top_p=self.config.top_p if top_p is None else top_p,
                    do_sample=True if (temperature or self.config.temperature) > 0 else False,
                )
                return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    def generate_messages(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        if getattr(self, "backend", None) == "gguf":
            # GGUF models use chat template for messages
            prompt = self._format_messages_gguf(messages)
            return self._generate_gguf(
                prompt,
                max_new_tokens or self.config.max_new_tokens,
                temperature if temperature is not None else self.config.temperature,
                top_p if top_p is not None else self.config.top_p
            )
        else:
            # Transformers backend: use Llama 3.1-style formatting
            with torch.inference_mode():
                # Use the consistent llama3-style formatter that preserves all system context
                prompt = self._format_messages_hf(messages)
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                    temperature=self.config.temperature if temperature is None else temperature,
                    top_p=self.config.top_p if top_p is None else top_p,
                    do_sample=True if (temperature or self.config.temperature) > 0 else False,
                )
                return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    def generate_messages_stream(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None):
        """Generate response with streaming tokens."""
        if getattr(self, "backend", None) == "gguf":
            # GGUF streaming
            prompt = self._format_messages_gguf(messages)
            yield from self._generate_gguf_stream(
                prompt,
                max_new_tokens or self.config.max_new_tokens,
                temperature if temperature is not None else self.config.temperature,
                top_p if top_p is not None else self.config.top_p
            )
        else:
            # Transformers backend streaming
            prompt = self._format_messages_hf(messages)
            yield from self._generate_hf_stream(
                prompt,
                max_new_tokens or self.config.max_new_tokens,
                temperature if temperature is not None else self.config.temperature,
                top_p if top_p is not None else self.config.top_p
            )

    def _generate_gguf(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        """Generate text using GGUF model with llama-cpp-python."""
        try:
            response = self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stop=self._gguf_stop_tokens,
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"GGUF generation failed: {e}")

    def _generate_gguf_stream(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
        """Generate text using GGUF model with streaming."""
        try:
            for chunk in self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stop=self._gguf_stop_tokens,
                stream=True,
            ):
                token = chunk["choices"][0]["text"]
                if token:
                    yield token
        except Exception:
            # Fallback to non-streaming
            response = self._generate_gguf(prompt, max_new_tokens, temperature, top_p)
            for char in response:
                yield char

    def _generate_hf_stream(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
        """Generate text using transformers backend with streaming."""
        with torch.inference_mode():
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Try to use HF TextIteratorStreamer if available
            try:
                from transformers import TextIteratorStreamer
                import threading
                
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    streamer=streamer,
                )
                
                thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                for token in streamer:
                    yield token
                    
                thread.join()
                
            except ImportError:
                # Fallback: generate full response then stream character by character
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                )
                response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                for char in response:
                    yield char

    def _format_messages_gguf(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for GGUF using Llama 3.1 Instruct format for compatibility."""
        parts: List[str] = ["<|begin_of_text|>"]
        
        # System messages first
        sys_msgs = [m.get("content", "") for m in messages if m.get("role") == "system"]
        system_prompt = "\n\n".join([s.strip() for s in sys_msgs if s.strip()])
        if system_prompt:
            parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")
        
        # Keep last 10 pairs (20 messages) for speed
        convo_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        if len(convo_msgs) > 20:
            convo_msgs = convo_msgs[-20:]
        
        # Format conversation
        i = 0
        while i < len(convo_msgs):
            msg = convo_msgs[i]
            if msg.get("role") == "user":
                user_content = msg.get("content", "").strip()
                parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>")
                
                # Check for assistant response
                if i + 1 < len(convo_msgs) and convo_msgs[i + 1].get("role") == "assistant":
                    asst_content = convo_msgs[i + 1].get("content", "").strip()
                    parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{asst_content}<|eot_id|>")
                    i += 2
                else:
                    # Incomplete pair - prepare for assistant response
                    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                    i += 1
            else:
                i += 1
        
        # Ensure we end ready for assistant response
        if not parts[-1].endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"):
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(parts)

    def _format_messages_hf(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using optimized Llama 3.1 Instruct format for transformers backend."""
        parts: List[str] = ["<|begin_of_text|>"]
        
        # System messages
        sys_msgs = [m.get("content", "") for m in messages if m.get("role") == "system"]
        system_prompt = "\n\n".join([s.strip() for s in sys_msgs if s.strip()])
        if system_prompt:
            parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")
        
        # Keep last 10 pairs (20 messages) for speed
        convo_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        if len(convo_msgs) > 20:
            convo_msgs = convo_msgs[-20:]
        
        # Fast sequential processing
        i = 0
        while i < len(convo_msgs):
            msg = convo_msgs[i]
            if msg.get("role") == "user":
                user_content = msg.get("content", "").strip()
                parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>")
                
                # Check for assistant response
                if i + 1 < len(convo_msgs) and convo_msgs[i + 1].get("role") == "assistant":
                    asst_content = convo_msgs[i + 1].get("content", "").strip()
                    parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{asst_content}<|eot_id|>")
                    i += 2
                else:
                    # No assistant response - prepare for generation
                    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                    break
            else:
                i += 1
        
        # Ensure we end with assistant header for generation
        if not parts[-1].endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"):
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(parts)