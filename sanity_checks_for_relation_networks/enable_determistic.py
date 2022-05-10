import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np  # noqa
import torch  # noqa

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)
