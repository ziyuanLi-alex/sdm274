
def set_seed(seed=42):
    """设置所有可能的随机种子以确保可重复性"""
    import numpy as np
    import random
    
    # Python的random模块
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    