import numpy as np
import pytest

try:
    from tools.hegre_dataset.review.procrustes import scale_and_center_landmarks
except ImportError:
    scale_and_center_landmarks = None


def create_mock_face():
    face = np.zeros((68, 2), dtype=np.float32)

    for i in range(36, 42):
        face[i] = [-10.0, 10.0]

    for i in range(42, 48):
        face[i] = [10.0, 10.0]

    face[30] = [0.0, 0.0]

    face[0] = [-20.0, -20.0]
    face[16] = [20.0, 20.0]

    return face


@pytest.mark.skipif(scale_and_center_landmarks is None,
                    reason="procrustes requires cv2 and scipy (heavy extras)")
def test_scale_and_center_landmarks():
    face = create_mock_face()

    face = face * 2.0 + np.array([50.0, 60.0])

    out_size = (300, 300)
    scaled = scale_and_center_landmarks(face, out_size)

    assert np.isclose(scaled[30][0], 150.0, atol=1e-5), "Nose tip X not centered!"
    assert np.isclose(scaled[30][1], 150.0, atol=1e-5), "Nose tip Y not centered!"
