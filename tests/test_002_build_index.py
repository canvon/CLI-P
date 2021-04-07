import tests.helper
tests.helper.setupWarningFilters()

import unittest
import re
from pathlib import Path

class TestBuildIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Call a second time to undo changes made by the test runner...
        tests.helper.setupWarningFilters()

        cls.path_prefix = "tests"
        cls.samples_path = "sample-images"
        cls.build_index = __import__("build-index")

    def test_0001_scan_nonrecursive_default(self):
        """Test that build-index run, with non-recursive default, will not find the sample images."""
        scanner = self.build_index.Scanner(path_prefix=self.path_prefix, clusters=1)
        output, errout, _ = tests.helper.capture_stdout_cstderr(lambda: scanner.run(self.samples_path))

        out_lines = output.splitlines()
        self.assertGreaterEqual(len(out_lines), 1, msg=f"Non-recursive Scanner didn't give any normal output. Error output was {errout!r}.")
        self.assertEqual(out_lines[-1], "Done!", msg=f"Non-recursive Scanner normal output didn't end in success. Output was {output!r}, error output was {errout!r}.")

        self.assertEqual(0, scanner.db.count_fn(), msg=f"Non-recursive Scanner found some images all-the-same. Output was {output!r}, error output was {errout!r}.")

    def test_scan_samples(self):
        """Test build-index run. This may take a long time, around 1-2 seconds per image."""
        # clusters: Would otherwise fail in faiss with:
        #
        #     Error: 'nx >= k' failed: Number of training points (12) should be
        #     at least as large as number of clusters (100)
        #
        scanner = self.build_index.Scanner(path_prefix=self.path_prefix, recursive=True, clusters=1)
        output, errout, _ = tests.helper.capture_stdout_cstderr(lambda: scanner.run(self.samples_path))
        # TODO: Check presence of output files, and that their timestamps
        #       are more recent, after run...

        err_whitelist = [
            r'WARNING clustering \d+ points to \d+ centroids: please provide at least \d+ training points',
        ]
        err_lines = errout.splitlines()
        for line in err_lines:
            found = False
            for pattern in err_whitelist:
                if re.match(pattern, line) is not None:
                    found = True
                    break
            self.assertTrue(found, msg=f"build-index had unexpected error output {line!r}. (All error output was {errout!r}. Output was {output!r}.)")

        dir = Path(self.samples_path)
        # (Go through file extensions, produce a recursive glob generator and,
        # from that, a constant 1 generator, which can then be summed up
        # to give the paths count without building up lists storing all the elements.
        # The per-extension counts are then summed up to give the overall result.)
        n_images_expected = sum(sum(1 for _ in dir.rglob('*' + ext)) for ext in scanner.file_extensions)
        n_images_db = scanner.db.count_fn()
        self.assertEqual(n_images_expected, n_images_db, msg=f"build-index gave {'less' if n_images_db < n_images_expected else 'more'} results than expected! Output was {output!r}.")

        out_lines = output.splitlines()
        self.assertGreaterEqual(len(out_lines), 2, msg=f"build-index didn't give enough output lines (but {output!r})")
        out_final = out_lines[-1]
        self.assertEqual("Done!", out_final, msg=f"build-index final output line didn't indicate success. Output was {output!r}.")
        out_summary = out_lines[-2]
        self.assertTrue(out_summary.startswith(f"Indexed {n_images_db} images"), msg=f"build-index didn't give expected summary line (but {out_summary!r}). Complete output was {output!r}.")

    def run(self, result=None):
        result = super().run(result)
        if len(result.errors) > 0 or len(result.failures) > 0:
            print(f"{self.__class__.__name__}: Running build-index failed, stopping test suite!"
                " (Further tests would just fail on missing files...)", flush=True)
            result.stop()

if __name__ == '__main__':
    unittest.main()
