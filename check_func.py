import numpy as np
from utility import vectorized_kick, vectorized_kick_batch, s_batch, s

def test_vectorized_kick_vs_batch():
    profiles = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    for i, profile in enumerate(profiles):
        expected = vectorized_kick(profile)
        batch_result = vectorized_kick_batch(profiles)[i]
        expected = np.array(expected)
        batch_result = np.array(batch_result)
        print(f"Profile {i}: Expected {expected}, Batch Result {batch_result}")
        assert np.allclose(expected, batch_result), f"Mismatch at profile {i}"
        
def test_s_vs_s_batch():
    profiles = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    s_individual = np.array([s(profile) for profile in profiles])
    s_batched = s_batch(profiles)
    print(f"s_individual: {s_individual}")
    print(f"s_batched: {s_batched}")
    assert np.allclose(s_individual, s_batched), "Mismatch between s and s_batch!"


if __name__ == "__main__":
    # test_vectorized_kick_vs_batch()
    test_s_vs_s_batch()
    print("All tests passed!")
    
    
    