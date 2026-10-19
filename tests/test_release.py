import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "normalize_sdist", Path(__file__).resolve().parents[1] / "scripts/normalize_sdist.py")
normalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalizer)


class ReleaseTests(unittest.TestCase):
    def test_archive_metadata_is_generic_and_contents_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                entry = tarfile.TarInfo("example/README.md")
                entry.size = 5
                entry.uid = entry.gid = 1234
                entry.uname = entry.gname = "example-user"
                entry.mtime = 123456
                archive.addfile(entry, io.BytesIO(b"hello"))
            normalizer.normalize(path)
            with tarfile.open(path) as archive:
                entry = archive.getmembers()[0]
                self.assertEqual((entry.uid, entry.gid, entry.uname, entry.gname, entry.mtime),
                                 (0, 0, "", "", 0))
                self.assertEqual(archive.extractfile(entry).read(), b"hello")
            first = path.read_bytes()
            normalizer.normalize(path)
            self.assertEqual(first, path.read_bytes())

    def test_unsafe_member_rejected_without_replacing_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.tar.gz"
            for name in ("../escape", "/absolute"):
                with tarfile.open(path, "w:gz") as archive:
                    archive.addfile(tarfile.TarInfo(name))
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    normalizer.normalize(path)
                self.assertEqual(before, path.read_bytes())

    def test_symlink_entry_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                entry = tarfile.TarInfo("example/link")
                entry.type = tarfile.SYMTYPE
                entry.linkname = "README.md"
                archive.addfile(entry)
            with self.assertRaises(ValueError):
                normalizer.normalize(path)
