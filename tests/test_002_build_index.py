import tests.helper
tests.helper.setupWarningFilters()

import unittest
from pathlib import Path

class TestBuildIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Call a second time to undo changes made by the test runner...
        tests.helper.setupWarningFilters()

        cls.path_prefix = "tests"
        cls.samples_path = "sample-images/downscale"
        cls.build_index = __import__("build-index")

    def test_scan_samples(self):
        # clusters: Would otherwise fail in faiss with:
        #
        #     Error: 'nx >= k' failed: Number of training points (12) should be
        #     at least as large as number of clusters (100)
        #
        scanner = self.build_index.Scanner(path_prefix=self.path_prefix, clusters=1)
        scanner.run(self.samples_path)
        # TODO: Check presence of output files, and that their timestamps
        #       are more recent, after run...
        dir = Path(self.samples_path)
        # (Go through file extensions, produce a glob generator and,
        # from that, a constant 1 generator, which can then be summed up
        # to give the paths count without building up lists storing all the elements.
        # The per-extension counts are then summed up to give the overall result.)
        n_images_expected = sum([sum(1 for _ in dir.glob('*' + ext)) for ext in scanner.file_extensions])
        with scanner.db.env.begin(db=scanner.db.fn_db) as txn:
            n_images_db = txn.stat()['entries']
        self.assertEqual(n_images_expected, n_images_db, msg=f"build-index gave {'less' if n_images_db < n_images_expected else 'more'} results than expected")

    def run(self, result=None):
        result = super().run(result)
        if len(result.errors) > 0 or len(result.failures) > 0:
            print(f"{self.__class__.__name__}: Running build-index failed, stopping test suite!"
                " (Further tests would just fail on missing files...)", flush=True)
            result.stop()

if __name__ == '__main__':
    unittest.main()
