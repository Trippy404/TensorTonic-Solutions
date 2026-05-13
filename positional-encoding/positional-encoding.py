import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
     # Create position indices (shape: seq_len, 1)
    positions = np.arange(seq_len).reshape(-1, 1)
    # Create dimension indices
    dims = np.arange(d_model)
    # Compute angle rates
    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))
    # Compute angle matrix
    angles = positions * angle_rates
    # Initialize positional encoding matrix
    pe = np.zeros((seq_len, d_model))
    # Apply sin to even indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe