from src.data import encode, decode


def test_encode():
    s = "0123456789+= "
    result = [n for n in range(13)]
    assert result == encode(s)
