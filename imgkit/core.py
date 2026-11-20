"""imgkit 核心 API：滤镜链（卷积核链）。"""

import threading

__all__ = ["FilterChain"]

_SUM_TOL = 1e-6
_EDGE_MODES = ("clamp", "zero")


def _validate_kernel(kernel):
    """校验核：非空、奇数方阵、数值、权重和为 1。非法抛 ValueError。"""
    if not isinstance(kernel, (list, tuple)) or len(kernel) == 0:
        raise ValueError("kernel 必须是非空方阵")
    n = len(kernel)
    if n % 2 == 0:
        raise ValueError("kernel 边长必须为奇数")
    total = 0.0
    for row in kernel:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError("kernel 必须是方阵")
        for v in row:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError("kernel 元素必须是数值")
            total += float(v)
    if abs(total - 1.0) > _SUM_TOL:
        raise ValueError("kernel 权重和必须为 1")
    return [[float(v) for v in row] for row in kernel]


def _convolve(image, kernel, edge):
    """单核卷积，返回新图，不修改输入。

    edge='zero'：边界外补 0，全图卷积。
    edge='clamp'：核覆盖到界外的边缘像素保持原值，仅内部像素卷积。
    """
    h = len(image)
    w = len(image[0]) if h else 0
    k = len(kernel)
    r = k // 2

    def at(y, x):
        if 0 <= y < h and 0 <= x < w:
            return image[y][x]
        return 0.0  # zero padding

    out = []
    for y in range(h):
        row = []
        for x in range(w):
            if edge == "clamp" and not (r <= y < h - r and r <= x < w - r):
                row.append(image[y][x])  # 边缘像素钳制为原值
                continue
            acc = 0.0
            for ky in range(k):
                for kx in range(k):
                    acc += kernel[ky][kx] * at(y + ky - r, x + kx - r)
            row.append(acc)
        out.append(row)
    return out


class FilterChain:
    def __init__(self):
        """构造空链"""
        self._kernels = []  # [(kernel, edge), ...]
        self._lock = threading.Lock()

    def add_kernel(self, kernel, edge="clamp"):
        """追加一个卷积核；非法输入抛 ValueError 且链保持不变"""
        if edge not in _EDGE_MODES:
            raise ValueError(f"未知边缘模式: {edge!r}")
        k = _validate_kernel(kernel)  # 先校验，失败则不触碰链
        with self._lock:
            self._kernels.append((k, edge))
        return self

    def rollback(self):
        """撤销最后一次追加；空链抛 ValueError"""
        with self._lock:
            if not self._kernels:
                raise ValueError("空链无法回退")
            self._kernels.pop()

    def apply(self, image):
        """整图处理；空链返回原图副本，不修改输入"""
        with self._lock:
            chain = list(self._kernels)  # 快照，保证并发/回退不影响进行中的 apply
        out = [list(row) for row in image]
        for kernel, edge in chain:
            out = _convolve(out, kernel, edge)
        return out

    def apply_stream(self, rows):
        """按行流式处理；结果与 apply 逐像素一致"""
        return self.apply([list(row) for row in rows])

    def kernels(self):
        """返回当前核列表（副本）"""
        with self._lock:
            return [k for k, _ in self._kernels]
