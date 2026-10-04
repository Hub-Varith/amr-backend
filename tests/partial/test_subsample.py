import gzip
import tempfile
import unittest
from pathlib import Path

from genome2mic.partial.subsample import subsample


class SubsampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "reads.fastq"
        self.source.write_text("".join(f"@r{i}\nACGT\n+\nIIII\n" for i in range(100)))

    def test_nested_reproducible_and_complete(self):
        first = subsample(self.source, self.root / "a", [.1, .5, 1.0])
        second = subsample(self.source, self.root / "b", [.1, .5, 1.0])
        sets = []
        for entry in first["subsets"]:
            a = (self.root / "a" / entry["file"]).read_bytes()
            self.assertEqual(a, (self.root / "b" / entry["file"]).read_bytes())
            sets.append(set(a.decode().splitlines()[::4]))
            self.assertEqual(entry["bases"], entry["reads"] * 4)
        self.assertTrue(sets[0] <= sets[1] <= sets[2])
        self.assertEqual((self.root / "a/subset_02.fastq").read_bytes(), self.source.read_bytes())
        self.assertEqual(first, second)

    def test_prefix_and_gzip(self):
        zipped = self.root / "reads.fastq.gz"
        with gzip.open(zipped, "wb") as handle:
            handle.write(self.source.read_bytes())
        result = subsample(zipped, self.root / "prefix", [.1, 1.0], mode="prefix")
        self.assertEqual([e["reads"] for e in result["subsets"]], [10, 100])
        self.assertEqual((self.root / "prefix/subset_00.fastq").read_text(),
                         "".join(self.source.read_text().splitlines(keepends=True)[:40]))

    def test_invalid_input_leaves_no_output(self):
        self.source.write_text("@r\nACGT\n+\nII\n")
        with self.assertRaises(ValueError):
            subsample(self.source, self.root / "bad", [.1])
        self.assertFalse((self.root / "bad").exists())

    def test_invalid_fractions_and_overwrite_refused(self):
        for fractions in [[], [0], [1.1], [float("nan")]]:
            with self.assertRaises(ValueError):
                subsample(self.source, self.root / "bad", fractions)
        with self.assertRaises(FileExistsError):
            subsample(self.source, self.root, [1.0])

    def test_empty_input(self):
        self.source.write_text("")
        with self.assertRaises(ValueError):
            subsample(self.source, self.root / "empty", [.1])


if __name__ == "__main__":
    unittest.main()
