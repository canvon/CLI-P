import os
import sys
import time
import logging
import warnings
import numpy as np
import cv2
import models_store  # (imports torch_device)
from torch_device import device as default_device
import torch
import torchvision.ops
import xdg.BaseDirectory
from torchvision import transforms
from align_faces import warp_and_crop_face
import align_faces
import main_helper

loggerName = main_helper.getLoggerName(name=__name__, package=__package__, file=__file__)
logger = logging.getLogger(loggerName)

def load_resnet(device=default_device):
    logger.info("Loading resnet model...")
    load_start = time.perf_counter()

    model = None
    from retinaface.pre_trained_models import get_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if device == 'cpu':
            model = get_model("resnet50_2020-07-20", max_size=720, device=device)
        else:
            model = get_model("resnet50_2020-07-20", max_size=1400, device=device)
    model.eval()

    load_time = time.perf_counter() - load_start
    logger.debug("Finished loading resnet model after %fs.", load_time)

    return model

RESNET_MODEL_KEY = "resnet"
store_resnet_model = models_store.store.register_lazy_or_getitem(RESNET_MODEL_KEY, load_resnet)

face_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def load_arcface(device=default_device):
    logger.info("Loading arcface model...")
    load_start = time.perf_counter()

    face_model = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(os.path.join(xdg.BaseDirectory.xdg_cache_home, "InsightFace-v2", "BEST_checkpoint_r101.tar"), map_location=torch.device(device))
        face_model = checkpoint['model'].module.to(device)
        face_model.device = device
    face_model.eval()

    load_time = time.perf_counter() - load_start
    logger.debug("Finished loading arcface model after %fs.", load_time)

    return face_model

ARCFACE_MODEL_KEY = "arcface"
store_arcface_model = models_store.store.register_lazy_or_getitem(ARCFACE_MODEL_KEY, load_arcface)

# image needs to be RGB not BGR
def embed_faces(annotations, filename=None, image=None):
    global face_model
    if len(annotations) < 1:
        return True
    with torch.no_grad():
        if image is None and filename is not None:
            image = cv2.imread(filename)
            if image is None:
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            return False
        face_model = store_arcface_model.get()
        device = store_arcface_model.loaded_device
        images = torch.zeros(len(annotations), 3, 112, 112).to(device)
        for i, annotation in enumerate(annotations):
            face = warp_and_crop_face(image, annotation['landmarks'], reference_pts=align_faces.REFERENCE_FACIAL_POINTS_112, crop_size=(112,112))
            images[i] = face_transforms(face).to(device)
        embedding = face_model(images)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()
        for i, annotation in enumerate(annotations):
            annotation['embedding'] = embedding[i]
        return True

def reorient(w, h, points, orientation):
    if orientation is None or orientation == 1:
        return points
    if len(points) == 4:
        a, b = reorient(w, h, (points[0], points[1]), orientation)
        c, d = reorient(w, h, (points[2], points[3]), orientation)
        return (min(a,c), min(b,d), max(a,c), max(b,d))
    if orientation == 3:
        # 180 degrees
        return (w - points[0], h - points[1])
    elif orientation == 6:
        # 270 degrees
        return (w - points[1], points[0])
    elif orientation == 8:
        # 90 degrees
        return (points[1], h - points[0])
    return points

def annotate(annotations, filename=None, image=None, scale=1.0, face_id=None, orientation=None, skip_landmarks=False):
    if image is None and filename is not None:
        image = cv2.imread(filename)
    if image is None:
        return None
    for i, annotation in enumerate(annotations):
        h, w, _ = image.shape
        color = (0, 0, 255)
        if face_id is not None and face_id == i:
            color = (0, 255, 0)
        if 'color' in annotation:
            color = annotation['color']
        bbox = (int(annotation['bbox'][0] * scale + 0.5), int(annotation['bbox'][1] * scale + 0.5), int(annotation['bbox'][2] * scale + 0.5), int(annotation['bbox'][3] * scale + 0.5))
        bbox = reorient(w, h, bbox, orientation)
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        if not skip_landmarks:
            for landmark in annotation['landmarks']:
                landmark = reorient(w, h, (int(landmark[0] * scale + 0.5), int(landmark[1] * scale + 0.5)), orientation)
                image = cv2.circle(image, landmark, 1, (0, 0, 255), 2)

        y = bbox[1] - 6
        if y - 16 < 0:
            y = bbox[3] + 16

        tag = ""
        if 'tag' in annotation:
            tag = " " + annotation['tag']

        image = cv2.putText(image, str(i) + tag, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        image = cv2.putText(image, str(i) + tag, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return image

# image needs to be RGB not BGR
def get_faces(filename=None, image=None):
    with torch.no_grad():
        if image is None and filename is not None:
            image = cv2.imread(filename)
            if image is None:
                return []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            return []
        model = store_resnet_model.get()
        annotations = model.predict_jsons(image, nms_threshold=0.25, confidence_threshold=0.85)
        if len(annotations) < 1 or len(annotations[0]['bbox']) != 4 or len(annotations) > 500:
            return []

        boxes = torch.zeros(len(annotations), 4)
        for i, annotation in enumerate(annotations):
            boxes[i, :] = torch.tensor((annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][2], annotation['bbox'][3])).float()

        valid_boxes = torchvision.ops.remove_small_boxes(boxes, 40)

        valid_annotations = []
        for box_id in valid_boxes:
            annotation = annotations[box_id]
            valid_annotations.append(annotation)
        embed_faces(valid_annotations, image=image)

        return valid_annotations

if __name__ == "__main__":
    store_arcface_model.get()
    for filename in sys.argv[1:]:
        process_start = time.perf_counter()
        annotations = get_faces(filename)
        process_time = time.perf_counter() - process_start
        print(f"Processing time: {process_time:.4f}s")
        image = annotate(annotations, filename)
        try:
            cv2.imshow('Image', image)
            cv2.waitKey(0)
        except:
            pass
