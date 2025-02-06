import numpy as np

def deep_getattr(struct, field_path):
    fields = field_path.split('.')
    cur = struct
    for field in fields:
        cur = getattr(cur, field)
    return cur


def compress_matrix(matrix, r):
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :r], np.diag(S[:r]), VT[:r, :]


def compress_matrix_full(matrix, r):
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]


class Positions:
    def __init__(self, model, way_to_layer, paths, ranks):
        self.way_to_layer = way_to_layer
        self.names = dict(zip(paths, ranks))
        self.data = {}
        self.model = model

    def get_lay(self, lay):
        return deep_getattr(self.model, self.way_to_layer)[lay]

    def get(self, lay, name):
        return deep_getattr(self.get_lay(lay), name)

