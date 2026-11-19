"""imgkit 核心 API（初始未实现）。"""


class FilterChain:
    def __init__(self):
        """构造空链"""
        raise NotImplementedError

    def add_kernel(self, kernel, edge='clamp'):
        """追加一个卷积核"""
        raise NotImplementedError

    def rollback(self):
        """撤销最后一次追加"""
        raise NotImplementedError

    def apply(self, image):
        """整图处理"""
        raise NotImplementedError

    def apply_stream(self, rows):
        """按行流式处理"""
        raise NotImplementedError

    def kernels(self):
        """返回当前核列表"""
        raise NotImplementedError
