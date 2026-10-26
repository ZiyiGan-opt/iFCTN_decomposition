import scipy.io
import numpy as np
import os


def load_data(filename):
    """加载 .mat 数据"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    try:
        data = scipy.io.loadmat(filename)
        # 自动寻找最大的变量作为数据
        for key in data:
            if not key.startswith('__') and isinstance(data[key], np.ndarray):
                return data[key]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None