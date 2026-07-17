from src.data import encode, decode


def test_encode():
    s = "0123456789+= "
    result = [n for n in range(13)]
    assert encode(s) == result


def test_decode():
    nums = [n for n in range(13)]
    result = "0123456789+="
    assert decode(nums) == result


def test_decode_strips():
    nums = [n for n in range(13)] + [12] * 5
    result = "0123456789+="
    assert decode(nums) == result


def test_encode_decode_inverse():
    s = "0123456789+= "
    assert decode(encode(s)) == s[:-1]
