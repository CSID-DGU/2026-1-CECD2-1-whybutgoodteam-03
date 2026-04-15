import sys
print("1. Numpy 임포트 시도...")
import numpy as np
print(f"   -> 성공! 버전: {np.__version__}")

print("2. Torch 임포트 시도...")
import torch
print("   -> 성공!")

print("3. TensorFlow 임포트 시도...")
import tensorflow as tf
print("   -> 성공!")

print("✅ 모든 라이브러리 정상")
