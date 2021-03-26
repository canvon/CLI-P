from pathlib import Path
import sys
import collections
import logging
import numpy as np
from torch_device import device
import torch
import clip
import faiss
from PIL import Image

import database
from faces import get_faces as get_face_embeddings, load_arcface

logger = logging.getLogger(__name__)


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


class Scanner:

    def __init__(self, *, faces=faces, pack_type=pack_type, clusters=clusters,
        file_extensions=file_extensions, skip_paths=skip_paths,
        path_prefix=None, loud=False, dry_run=False):
        # Copy instance variables from keyword arguments defaulted to globals.
        self.faces = faces
        self.pack_type = pack_type
        self.clusters = clusters
        self.file_extensions = list(file_extensions)
        self.skip_paths = list(skip_paths)
        if path_prefix is None:
            path_prefix = Path('.')
        elif type(path_prefix) is str:
            path_prefix = Path(path_prefix)
        self.path_prefix = path_prefix
        self.loud = loud
        self.dry_run = dry_run

        if self.dry_run:
            print("Dry run: Skipping load CLIP model.")
        else:
            self.model, self.transform = clip.load("ViT-B/32", device=device, jit=False)
            self.model.eval()

        if self.faces:
            if self.dry_run:
                print("Dry run: Additionally, would have loaded arcface, now.")
            else:
                load_arcface()

        self.db = database.get(path_prefix=self.path_prefix, pack_type=self.pack_type)

    @torch.no_grad()
    def clip_file(self, fn):
        fn = Path(fn)  # Upgrade potential string. Should be harmless when already a Path instance.
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
                image = self.transform(image).unsqueeze(0).to(device)
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
                logger.warning("build-index.Scanner.clip_file(): Exception %s processing image %r%s: %s",
                    type(ex).__name__, tfn, " (will mark as skip in db)" if mark_skip else "", ex)
            if mark_skip:
                self.db.put_skip(tfn)
            return False

    def clip_paths(self, *base_paths, sort_fns=False):
        path_queue = collections.deque(map(Path, base_paths))  # map(): Upgrade potential strings. Should be harmless when already a Path instance.
        while len(path_queue) > 0:
            path = path_queue.popleft()
            print(f"CLIPing {str(path)!r}...")  # (Quote a bit, but not too much. User output would not need PosixPath(...) wrapping.)
            fns = None
            if not path.is_dir():
                # Treat explicitly given filename as directory with a single entry.
                # Should help to share code, and to support them at all.
                fns = [path]
            elif not sort_fns:
                # Efficient: Don't store intermediate list of all files in directory.
                fns = path.iterdir()
            else:
                # Meaningful database index numbers, resembling directory listed by ls or a file manager.
                fns = list(path.iterdir())
                fns.sort()
            # Iterate through all the dir's filenames.
            subdirs = []
            for fn in fns:
                # Collect sub-directories for later front-queueing.
                if fn.is_dir():
                    subdirs.append(fn)
                    continue
                result = self.clip_file(fn)
                if result is None:
                    # Indicates a skip. Don't output anything.
                    if self.dry_run:
                        # (..except for dry run.)
                        print("_", end="", flush=True)
                    continue
                if result:
                    # Indicate successful processing of image into the database.
                    print(".", end="", flush=True)
                else:
                    # Indicate error.
                    print("#", end="", flush=True)
            print(flush=True)
            # Front-queue collected directories.
            if sort_fns:
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
            print("Dry run: This will visit all paths and load all images; but AI processing will be skipped.")
        try:
            self.clip_paths(*base_paths)
        except KeyboardInterrupt:
            print(f"Interrupted!")

        self.index_images()


if __name__ == '__main__':
    scanner = Scanner()
    scanner.run(*sys.argv[1:])  # (Unpack command-line arguments (except for script name) as positional parameters.)
