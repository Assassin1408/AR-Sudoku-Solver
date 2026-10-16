"""Download the raw MNIST idx files into data/MNIST/raw (tries several mirrors).

    python download_mnist.py
"""
import gzip
import os
import ssl
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "MNIST", "raw")
FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
           "https://storage.googleapis.com/cvdf-datasets/mnist/",
           "https://raw.githubusercontent.com/fgnt/mnist/master/"]


def ssl_context():
    """Python from python.org on macOS ships without system certificates,
    which breaks HTTPS downloads. certifi's bundle fixes that everywhere."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def main(root=DATA_DIR):
    os.makedirs(root, exist_ok=True)
    ctx = ssl_context()
    for f in FILES:
        dst = os.path.join(root, f)
        if os.path.exists(dst):
            print("already have", f)
            continue
        for m in MIRRORS:
            try:
                req = urllib.request.Request(m + f + ".gz", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                    data = gzip.decompress(r.read())
                with open(dst, "wb") as out:
                    out.write(data)
                print("got", f, "from", m)
                break
            except Exception as e:  # try the next mirror
                print("mirror failed:", m, type(e).__name__, e)
        else:
            raise SystemExit(f"Could not download {f}. Check your internet connection, "
                             f"or place the file manually in {root}")
    print("MNIST ready in", root)


if __name__ == "__main__":
    main()
