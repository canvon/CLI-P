# N.B.: Make sure to import this module only after Qt has been initialized
# in your code (if you do so), or there will be weird conflicts / start-up errors..:
#
#     QObject::moveToThread: Current thread (0x...) is not the object's thread (0x...).
#     Cannot move to target thread (0x...)
#
#     qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
#
import cv2


def load_image(*, tfn, max_res):
    """
    Prior part of query-index.Search.prepare_image() that doesn't need access to
    a Search instance or full Result record.

    This may be used for thumbnail loading.
    """
    image = cv2.imread(tfn, cv2.IMREAD_COLOR)
    if image is None or image.shape[0] < 2:
        return None, None
    h, w, _ = image.shape
    scale = 1.0
    if max_res is not None:
        need_resize = False
        if w > max_res[0]:
            factor = float(max_res[0])/float(w)
            w = max_res[0]
            h *= factor
            need_resize = True
            scale *= factor
        if h > max_res[1]:
            factor = float(max_res[1])/float(h)
            h = max_res[1]
            w *= factor
            need_resize = True
            scale *= factor
        if need_resize:
            image = cv2.resize(image, (int(w + 0.5), int(h + 0.5)), interpolation=cv2.INTER_LANCZOS4)
    return image, scale
