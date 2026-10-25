"""Current hanguard classification training entrypoint; see docs/training.md.

Binary: Qwen3.5 + LoRA + MLP/fusion. Category: frozen binary backbone +
five-class MLP/query readout. The retired generated-label SFT path is not used.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.hanguard.current_training import main


if __name__ == '__main__':
    main()
