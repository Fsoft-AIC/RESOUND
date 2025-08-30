from sma import StepwiseMonotonicMultiheadAttention

ref_attention = StepwiseMonotonicMultiheadAttention(256, 256//4, 256//4)
print("done")