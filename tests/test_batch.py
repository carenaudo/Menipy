import numpy as np
import pandas as pd

from menipy.batch import run_batch


def test_run_batch(tmp_path):
    cv2 = __import__('cv2')
    img1 = np.zeros((20, 20), dtype=np.uint8)
    cv2.circle(img1, (10, 10), 5, 255, -1)
    img2 = np.zeros((20, 20), dtype=np.uint8)
    cv2.rectangle(img2, (5, 5), (15, 15), 255, -1)
    path1 = tmp_path / 'a.png'
    path2 = tmp_path / 'b.png'
    cv2.imwrite(str(path1), img1)
    cv2.imwrite(str(path2), img2)

    output_csv = tmp_path / 'results.csv'
    df = run_batch(tmp_path, output_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert (df['area'] > 0).all()
    assert output_csv.exists()
