import re
import struct


def clean_ascii(text):
    return re.sub(r"[^\x20-\x7F]", ".", text)


def xxd(x, start=0, stop=None):
    if stop is None:
        stop = len(x)
    for i in range(start, stop, 8):
        # Row number
        print("%04d" % i, end="   ")
        # Hexadecimal bytes
        for r in range(i, i + 8):
            print("%02x" % x[r], end="")
            if (r + 1) % 4 == 0:
                print("  ", end="")
        # ASCII
        print(
            "   ",
            clean_ascii(x[i : i + 8].decode("utf-8", errors="ignore")),
            "   ",
            end="",
        )
        # Int32
        print(
            "{:>10} {:>10}".format(*struct.unpack("II", x[i : i + 8])),
            end="   ",
        )
        print("")  # Newline
    return


# Buffer reading functions
def read_int(buffer, n=1):
    res = struct.unpack("I" * n, buffer.read(4 * n))
    if n == 1:
        res = res[0]
    return res


def read_string(buffer):
    return "".join([x.decode() for x in iter(lambda: buffer.read(1), b"\x00")])


def read_delim(buffer, n):
    delim = read_int(buffer, n)
    assert all([x == 0 for x in delim]), "Unknown nonzero value in delimiter"
