import threading

import pytest

import imgkit

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

BOX = None


def _box():
    global BOX
    if BOX is None:
        BOX = [[1.0 / 9.0] * 3 for _ in range(3)]
    return BOX


def _img():
    return [[float((r * 5 + c) % 7) for c in range(5)] for r in range(5)]


def _chain():
    ch = imgkit.FilterChain()
    ch.add_kernel(_box())
    return ch


def _empty_identity():
    ch = imgkit.FilterChain()
    img = _img()
    out = ch.apply(img)
    return out == img and out is not img


def _rollback_restores():
    ch = _chain()
    before = ch.apply(_img())
    ch.add_kernel(_box())
    ch.rollback()
    return ch.apply(_img()) == before


def _rollback_empty():
    return _raises(lambda: imgkit.FilterChain().rollback())


def _stream_equals_apply():
    ch = _chain()
    img = _img()
    return ch.apply_stream(iter(img)) == ch.apply(img)


def _stream_accepts_generator():
    ch = _chain()
    img = _img()
    return ch.apply_stream((row for row in img)) == ch.apply(img)


def _bad_kernel_keeps_chain():
    ch = _chain()
    n = len(ch.kernels())
    ok = _raises(lambda: ch.add_kernel([[0.5, 0.5], [0.5, 0.5]]))
    return ok and len(ch.kernels()) == n


def _edge_modes_differ():
    img = [[0.0] * 3, [0.0, 9.0, 0.0], [0.0] * 3]
    a = imgkit.FilterChain()
    a.add_kernel(_box(), "clamp")
    b = imgkit.FilterChain()
    b.add_kernel(_box(), "zero")
    return a.apply(img) != b.apply(img)


def _kernels_listed():
    ch = _chain()
    ch.add_kernel(_box())
    return len(ch.kernels()) == 2


def _no_mutation():
    img = _img()
    snapshot = [row[:] for row in img]
    _chain().apply(img)
    return img == snapshot


def _shape_preserved():
    out = _chain().apply(_img())
    return len(out) == 5 and len(out[0]) == 5


def _threads_match():
    ch = _chain()
    img = _img()
    serial = ch.apply(img)
    out = {}
    threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, ch.apply(img))) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return all(out[i] == serial for i in range(4))


def _two_chains_independent():
    a = imgkit.FilterChain()
    a.add_kernel(_box())
    b = imgkit.FilterChain()
    return len(a.kernels()) == 1 and len(b.kernels()) == 0


def _rollback_twice():
    ch = _chain()
    ch.rollback()
    return _raises(lambda: ch.rollback()) and len(ch.kernels()) == 0


def _add_after_rollback():
    ch = _chain()
    ch.rollback()
    ch.add_kernel(_box())
    return len(ch.kernels()) == 1


def _unnormalized_rejected():
    return _raises(lambda: imgkit.FilterChain().add_kernel([[2.0]]))

CASES = [
        ("empty_identity", lambda: _empty_identity()),
        ("rollback", lambda: _rollback_restores()),
        ("rollback_empty", lambda: _rollback_empty()),
        ("stream", lambda: _stream_equals_apply()),
        ("stream_gen", lambda: _stream_accepts_generator()),
        ("bad_kernel", lambda: _bad_kernel_keeps_chain()),
        ("edge_modes", lambda: _edge_modes_differ()),
        ("kernels", lambda: _kernels_listed()),
        ("no_mutation", lambda: _no_mutation()),
        ("shape", lambda: _shape_preserved()),
        ("threads", lambda: _threads_match()),
        ("independent", lambda: _two_chains_independent()),
        ("rollback_twice", lambda: _rollback_twice()),
        ("add_after_rollback", lambda: _add_after_rollback()),
        ("unnormalized", lambda: _unnormalized_rejected()),
    ]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
