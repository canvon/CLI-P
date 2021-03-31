import sys
import time
from pathlib import Path
import collections
import logging
import numpy as np
import models_store  # (imports torch_device)
import torch
import clip
import faiss
from PIL import Image

import main_helper
import database
from faces import get_faces as get_face_embeddings

loggerName = main_helper.getLoggerName(name=__name__, package=__package__, file=__file__)
logger = logging.getLogger(loggerName)


# Enable to run face detection and calculate face embeddings that can be used to search for faces
faces = True

# If you have more than 2^32 images or faces, set this to '<Q'
pack_type = '<L'

# Split up index into this many clusters, 100 seems like a good number, but having at the very least 36 * clusters images is recommended
clusters = 100

# Accepted file extensions (have to be readable as standard RGB images by pillow and opencv)
file_extensions = ['.jpg', '.jpeg', '.png']

# Paths containing these will be skipped during index creation
skip_paths = []

# Path prefix to prepend before any database path names.
path_prefix = None

# Sort directory listings before processing the directory.
# This should give more predictable behaviour like the ls command or a file manager.
sort_fns = True

# Loud mode will spill out exceptions / log error messages.
loud = False

# Dry run will read all directories / image files, but skip AI processing.
# Used for development/testing.
dry_run = False


DRY_RUN_MSG = "This will visit all paths and load all images; but AI processing will be skipped."

CLIP_MODEL_KEY = "clip"
store_clip_model = models_store.store.register_lazy_or_getitem(CLIP_MODEL_KEY,
    lambda device: clip.load("ViT-B/32", device=device, jit=False))


class Scanner:

    def __init__(self, *, faces=faces, pack_type=pack_type, clusters=clusters,
        file_extensions=file_extensions, skip_paths=skip_paths,
        path_prefix=path_prefix, sort_fns=sort_fns, loud=loud, dry_run=dry_run):


        # Copy instance variables from keyword arguments defaulted to globals.

        if not isinstance(faces, bool):
            raise TypeError(f"faces needs to be a bool, but got type {type(faces)}")
        self.faces = faces

        if not isinstance(pack_type, str):
            raise TypeError(f"pack_type needs to be a string, but got type {type(pack_type)}")
        if pack_type not in ['<L', '<Q']:
            raise ValueError(f"invalid pack_type {pack_type!r}")
        self.pack_type = pack_type

        if not isinstance(clusters, int):
            raise TypeError(f"clusters needs to be an int, but got type {type(clusters)}")
        self.clusters = clusters

        # Validate file extensions list.
        if not isinstance(file_extensions, list):
            raise TypeError(f"file_extensions needs to be a list of strings, but got type {type(file_extensions)}")
        for i, ext in enumerate(file_extensions):
            if not isinstance(ext, str):
                raise TypeError(f"file_extensions needs to be a list of strings, but index {i} has type {type(ext)}")
            if not ext.startswith('.'):
                raise ValueError(f"file_extensions[{i}] doesn't start with a dot '.', but reads {ext!r}")
            if not len(ext) > 1:
                raise ValueError(f"file_extensions[{i}] doesn't contain an extension, but reads {ext!r}")
        self.file_extensions = list(file_extensions)  # (Take a copy in case the original changes, or we want to change our view.)

        # Validate skip paths list.
        if not isinstance(skip_paths, list):
            raise TypeError(f"skip_paths needs to be a list of strings, but got type {type(skip_paths)}")
        for i, path in enumerate(skip_paths):
            if not isinstance(path, str):
                raise TypeError(f"skip_paths needs to be a list of strings, but index {i} has type {type(path)}")
        self.skip_paths = list(skip_paths)  # (Take copy.)

        if path_prefix is None:
            path_prefix = Path('.')
        elif not isinstance(path_prefix, Path):
            path_prefix = Path(path_prefix)
        self.path_prefix = path_prefix

        if not isinstance(sort_fns, bool):
            raise TypeError(f"sort_fns needs to be a bool, but got type {type(sort_fns)}")
        self.sort_fns = sort_fns

        if not isinstance(loud, bool):
            raise TypeError(f"loud needs to be a bool, but got type {type(loud)}")
        self.loud = loud

        if not isinstance(dry_run, bool):
            raise TypeError(f"dry_run needs to be a bool, but got type {type(dry_run)}")
        self.dry_run = dry_run


        self.loaded_clip_model = False

        self.db = database.get(path_prefix=self.path_prefix, pack_type=self.pack_type)

    def load_clip_model(self):
        if self.loaded_clip_model:
            return
        if self.dry_run:
            logger.warning("Loading CLIP model during dry run... This should not happen!")
        logger.info("Loading CLIP model...")
        load_start = time.perf_counter()

        self.model, self.transform = store_clip_model.get()
        self.clip_device = store_clip_model.loaded_device
        self.model.eval()
        self.loaded_clip_model = True

        load_time = time.perf_counter() - load_start
        logger.debug("Finished loading CLIP model after %fs.", load_time)

    @torch.no_grad()
    def clip_file(self, fn):
        """
        CLIP a single file. Checks against several counter-indications, which return None.
        Any problems on actually processing the given file name are caught, and return False.
        Otherwise (or just after reading as image file in case of dry run), returns True.

        Note that this function will record previously-excepted file names in the database permanently
        (unless during dry run). Run clear_skip.py to reset that part of the db.

        :param fn: File name of image to CLIP; ideally, already a pathlib.Path instance.
        :returns: On skip, returns None. On problem processing image, returns False. Otherwise, returns True.
        """
        # Upgrade potential string. Keep unchanged when already a Path instance, due to time cost...
        if not isinstance(fn, Path):
            fn = Path(fn)
        tfn = str(fn)
        if fn.suffix.lower() not in self.file_extensions:
            return
        if self.db.check_skip(tfn):
            return
        clip_done = self.db.check_fn(tfn)
        faces_done = not self.faces or self.db.check_face(tfn)
        if clip_done and faces_done:
            return
        image = None
        try:
            image = Image.open(tfn).convert("RGB")
            if self.dry_run:
                return True
            idx = None
            if not faces_done:
                rgb = np.array(image)
            if not clip_done:
                self.load_clip_model()
                image = self.transform(image).unsqueeze(0).to(self.clip_device)
                image_features = self.model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                idx = self.db.put_fn(tfn, image_features.detach().cpu().numpy())
            else:
                idx = self.db.get_fn_idx(tfn)
            if not faces_done:
                annotations = get_face_embeddings(image=rgb)
                self.db.put_faces(idx, annotations)
            return True
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            mark_skip = True
            if self.dry_run:
                mark_skip = False
            if self.loud:
                logger.warning("Got exception %s processing image %r%s: %s",
                    type(ex).__name__, tfn, " (will mark as skip in db)" if mark_skip else "", ex)
            if mark_skip:
                self.db.put_skip(tfn)
            return False

    def clip_paths(self, *base_paths):
        PATH_EXCEPTION_WHITELIST = (FileNotFoundError, PermissionError)
        path_queue = collections.deque(
            # Upgrade potential strings. Use unchanged when already a Path instance, due to time cost...
            map(lambda p: p if isinstance(p, Path) else Path(p), base_paths)
        )
        while len(path_queue) > 0:
            path = path_queue.popleft()
            print(f"CLIPing {str(path)!r}...")  # (Quote a bit, but not too much. User output would not need PosixPath(...) wrapping.)
            subdirs = []
            try:
                fns = None
                if not path.is_dir():
                    # Treat explicitly given filename as directory with a single entry.
                    # Should help to share code, and to support them at all.
                    fns = [path]
                elif not self.sort_fns:
                    # Efficient: Don't store intermediate list of all files in directory.
                    fns = path.iterdir()
                else:
                    # Meaningful database index numbers, resembling directory listed by ls or a file manager.
                    fns = list(path.iterdir())
                    fns.sort()
                # Iterate through all the dir's filenames.
                for fn in fns:
                    fn_visual = '?'
                    # (Both ensure fn_visual will be output, as well as that
                    # a single filename error won't prevent processing of
                    # rest of directory (and collecting all the sub-directories).)
                    try:
                        # Collect sub-directories for later front-queueing.
                        if fn.is_dir():
                            fn_visual = '/'
                            subdirs.append(fn)
                            continue

                        # Actually process file as image.
                        result = self.clip_file(fn)

                        # Decide status indicator for this step in the long-running operation.
                        if result is None:
                            fn_visual = '_' if self.dry_run else None
                            continue
                        fn_visual = '.' if result else '#'
                    except PATH_EXCEPTION_WHITELIST as ex:
                        if self.loud:
                            logger.warning("Got exception %s while processing path element %r: %s",
                                type(ex).__name__, str(fn), ex)
                        fn_visual = '#'
                        continue
                    finally:
                        if fn_visual is not None:
                            print(fn_visual, end="", flush=True)
            except PATH_EXCEPTION_WHITELIST as ex:
                if self.loud:
                    logger.warning("Got exception %s while iterating through path %r: %s",
                        type(ex).__name__, str(path), ex)
                print("!", end="", flush=True)
                continue
            finally:
                print(flush=True)
                # Front-queue collected directories.
                if self.sort_fns:
                    subdirs.sort()
                path_queue.extendleft(subdirs)

    def index_images(self):
        if self.dry_run:
            print("Dry run: Skipping prepare indexes.")
            return
        i = 0
        faces_i = 0
        with self.db.env.begin(db=self.db.fn_db) as fn_txn:
            n = fn_txn.stat()['entries']
            with self.db.env.begin(db=self.db.fix_idx_db) as txn:
                nd = n
                count = 0
                faces_count = 0
                need_training = True
                faces_need_training = True
                if nd > 32768:
                    nd = 32768
                cursor = fn_txn.cursor()
                if cursor.first():
                    if self.faces:
                        faces_array = np.zeros((nd, 512))
                    images = np.zeros((nd, 512))
                    print(f"Preparing indexes...")
                    quantizer = faiss.IndexFlatIP(512)
                    index = faiss.IndexIVFFlat(quantizer, 512, self.clusters, faiss.METRIC_INNER_PRODUCT)
                    if self.faces:
                        faces_quantizer = faiss.IndexFlatIP(512)
                        faces_index = faiss.IndexIVFFlat(faces_quantizer, 512, self.clusters, faiss.METRIC_INNER_PRODUCT)
                    print(f"Generating matrix...")
                    for fn_hash, fix_idx in cursor:
                        fn = txn.get(fix_idx + b'n').decode()
                        skip = False
                        for skip_path in self.skip_paths:
                            if skip_path in fn:
                                skip = True
                                break
                        if skip:
                            continue
                        v = self.db.get_fix_idx_vector(self.db.b2i(fix_idx)).reshape((512,))
                        images[count, :] = v
                        count += 1
                        self.db.put_idx(i, fix_idx)
                        i += 1
                        if count == nd:
                            count = 0
                            images = images.astype('float32')
                            if need_training:
                                print(f"Training index {images.shape}...")
                                index.train(images)
                                need_training = False
                            print(f"Adding to index...")
                            index.add(images)
                            images = np.zeros((nd, 512))
                        if self.faces:
                            annotations = self.db.get_faces(fix_idx)
                            for face_idx, annotation in enumerate(annotations):
                                faces_array[faces_count, :] = annotation['embedding'][0].reshape((512,))
                                faces_count += 1
                                self.db.put_idx_face(faces_i, fix_idx, face_idx)
                                faces_i += 1
                                if faces_count == nd:
                                    faces_count = 0
                                    faces_array = faces_array.astype('float32')
                                    if faces_need_training:
                                        print(f"Training faces index {faces_array.shape}...")
                                        faces_index.train(faces_array)
                                        faces_need_training = False
                                    print(f"Adding to faces index...")
                                    faces_index.add(faces_array)
                                    faces_array = np.zeros((nd, 512))
                    if count > 0:
                        images = images[0:count].astype('float32')
                        if need_training:
                            print(f"Training index {images.shape}...")
                            index.train(images)
                        print(f"Adding to index...")
                        index.add(images)
                    if faces_count > 0:
                        faces_array = faces_array[0:faces_count].astype('float32')
                        if faces_need_training:
                            print(f"Training faces index {faces_array.shape}...")
                            faces_index.train(faces_array)
                        print(f"Adding to faces index...")
                        faces_index.add(faces_array)
                    print(f"Saving index...")
                    faiss.write_index(index, str(self.path_prefix / "images.index"))
                    if faces:
                        print(f"Saving faces index...")
                        faiss.write_index(faces_index, str(self.path_prefix / "faces.index"))

        print(f"Indexed {i} images and {faces_i} faces.")
        print(f"Done!")

    def run(self, *base_paths):
        if self.dry_run:
            print("Dry run:", DRY_RUN_MSG)
        try:
            self.clip_paths(*base_paths)
        except KeyboardInterrupt:
            print(f"Interrupted!")

        self.index_images()


if __name__ == '__main__':
    parser = main_helper.CLI.createArgvParser(description="Indexes images for CLI-P.")

    parser.add_argument('base_paths', nargs='*', type=Path, metavar='BASE_PATH',
        help="Filesystem paths to CLIP."
            " Will recurse through directories."
            " Image file names can be given explicitly,"
            " but still must match one of the configured file extensions."
            "\n\nWhen no base paths are given, will only rebuild the faiss index.")

    BOOL_DEFAULT_MSG = "(This is the default.)"

    parser.add_argument('--faces', dest='faces', default=faces, action='store_const', const=True,
        help="Run face detection and calculate face embeddings that can be used to search for faces." +
            (" " + BOOL_DEFAULT_MSG if faces else ""))
    parser.add_argument('--no-faces', dest='faces', action='store_const', const=False,
        help=("" if faces else BOOL_DEFAULT_MSG))

    parser.add_argument('--pack-type', default=pack_type,
        help="If you have more than 2^32 images or faces, set this to '<Q' instead of '<L'." +
            f" (The default is {pack_type!r}.)")

    parser.add_argument('--clusters', type=int, default=clusters,
        help="Split up index into this many clusters, 100 seems like a good number,"
            " but having at the very least 36 * clusters images is recommended." +
            f" (The default is {clusters}.)")

    parser.add_argument('--file-extensions', dest='file_extensions', metavar='EXTS',
        type=(lambda exts: exts.split(',')), default=list(file_extensions),  # (Note: Take copy!)
        help="Accepted file extensions. (Have to be readable as standard RGB images by pillow and opencv.)"
            "\n\nSpecify as comma-separated list, with leading dot on each extension." +
            f" (The default is {','.join(file_extensions)!r}.)")
    parser.add_argument('--add-file-extension', dest='file_extensions', metavar='EXT', action='append',
        help="Adds a single file extension to the list of accepted file extensions."
            " Specify with leading dot.")

    parser.add_argument('--skip-paths', dest='skip_paths', metavar='PATHS',
        type=(lambda paths: paths.split(',')), default=list(skip_paths),  # (Note: Take copy!)
        help="Paths containing these will be skipped during index creation."
            " Specify as comma-separated list." +
            f" (The default is {','.join(skip_paths)!r}.)")
    parser.add_argument('--add-skip-path', dest='skip_paths', metavar='PATH', action='append',
        help="Adds a single path to the list of to-be-skipped paths.")

    parser.add_argument('--path-prefix', type=Path, default=path_prefix,
        help="Path prefix to prepend before any database path names." +
            f" (The default is {path_prefix!r}, but the Path instance around argument will automatically be added.)")

    parser.add_argument('--sort-filenames', '--sort-fns', dest='sort_fns', default=sort_fns,
        action='store_const', const=True,
        help="Sort directory listings before processing the directory."
            " This should give more predictable behaviour like the ls command or a file manager." +
            (" " + BOOL_DEFAULT_MSG if sort_fns else ""))
    parser.add_argument('--no-sort-filenames', '--no-sort-fns', dest='sort_fns',
        action='store_const', const=False,
        help=("" if sort_fns else BOOL_DEFAULT_MSG))

    parser.add_argument('--loud', dest='loud', default=loud, action='store_const', const=True,
        help="Log exception message instead of just printing a status character. ('#' or '!')" +
            ("" + BOOL_DEFAULT_MSG if loud else ""))
    parser.add_argument('--no-loud', dest='loud', action='store_const', const=False,
        help=("" if loud else BOOL_DEFAULT_MSG))

    parser.add_argument('--dry-run', dest='dry_run', default=dry_run, action='store_const', const=True,
        help=DRY_RUN_MSG +
            (" " + BOOL_DEFAULT_MSG if dry_run else ""))
    parser.add_argument('--no-dry-run', dest='dry_run', action='store_const', const=False,
        help=("" if dry_run else BOOL_DEFAULT_MSG))

    cli = main_helper.setupCLI(argvParser=parser)

    try:
        scanner = Scanner(
            faces=cli.args.faces,
            pack_type=cli.args.pack_type,
            clusters=cli.args.clusters,
            file_extensions=cli.args.file_extensions,
            skip_paths=cli.args.skip_paths,
            path_prefix=cli.args.path_prefix,
            sort_fns=cli.args.sort_fns,
            loud=cli.args.loud,
            dry_run=cli.args.dry_run,
        )
    # Simplify error output for likely user / command-line argument errors:
    except (TypeError, ValueError) as ex:
        logger.fatal("Got exception %s from build-index.Scanner constructor (likely a command-line argument error): %s",
            type(ex).__name__, ex)
        sys.exit(2)

    scanner.run(*cli.args.base_paths)
