# Stable Audio 3 (MLX runtime), vendored

From https://github.com/Stability-AI/stable-audio-3 `optimized/mlx` (MIT License, see LICENSE): the generation
script and model definitions only (no installer, training, Gradio or Ableton parts).

Change: `scripts/weights.py` reads `SA3_HOME` so weights live in the app's data folder.

The model weights (`stabilityai/stable-audio-3-optimized`, Small SFX + T5Gemma + SAME-S decoder, ~1.7 GB) are
downloaded on first use and are NOT covered by the MIT License: they are under the Stability AI Community License
(free for individuals and organisations under USD 1M annual revenue; https://stability.ai/license) and T5Gemma under
the Gemma Terms of Use. Sounds made with it are labelled with that license in the sound library.
