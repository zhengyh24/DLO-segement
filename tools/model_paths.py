"""Portable paths for externally installed foundation models."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SAM31_REPO = Path(os.environ.get('SAM31_REPO', str(ROOT/'external/sam3')))
SAM31_CHECKPOINT = Path(os.environ.get('SAM31_CHECKPOINT', str(ROOT/'weights/sam3.1_multiplex_fp16.pt')))
HQ_CHECKPOINT = Path(os.environ.get('HQ_CHECKPOINT', str(ROOT/'weights/sam_hq_vit_b.pth')))
