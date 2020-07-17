import cv2
import numpy as np
from PIL import Image


def save_image(im, filename):
    im = Image.fromarray(im)
    im.save(filename)


def get_contours(im, min_area, max_area):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hulls = get_contours_gray(im, min_area, max_area)
    return hulls


def get_contours_gray(im, min_area, max_area):
    # im = cv2.blur(im, (3, 3))
    ret, thresh = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    contours = [c for c in contours if cv2.contourArea(c) < max_area]
    hulls = []
    for i in range(len(contours)):
        hulls.append(cv2.convexHull(contours[i], False))
    return hulls


def get_lines(edges_im):
    lines = cv2.HoughLinesP(
        image=edges_im,
        rho=50,
        theta=1 * np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10)
    return lines


def draw_lines(im, lines):
    color = (0, 255, 0)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(im, (x1, y1), (x2, y2), color, 2)
    return im


def pred2im(im_set, dsize, image_idx):
    im = np.zeros((dsize[0], dsize[1], 3))
    im[:, :, 0] = im_set[image_idx, :, :, 0]
    im[:, :, 1] = im_set[image_idx, :, :, 0]
    im[:, :, 2] = im_set[image_idx, :, :, 0]
    im = im.astype('uint8')
    return im


def image_set2list(y_train_pred, y_val_pred):
    images = []
    for im_set in [y_train_pred, y_val_pred]:
        dsize = im_set.shape[1:3]
        for image_idx in range(im_set.shape[0]):
            im = pred2im(im_set, dsize, image_idx)
            images.append(im)
    return images


def get_min_area_rectangle(contours):
    contour = contours[0]
    rectangle = cv2.minAreaRect(contour)
    rectangle = cv2.boxPoints(rectangle)
    rectangle = np.int0(rectangle)
    return rectangle


def get_xs(rectangle):
    c = np.median(rectangle, axis=0)
    result = None
    if rectangle is not None:
        x01 = []
        x23 = []
        for point_idx in range(len(rectangle)):
            point = rectangle[point_idx]
            if point[0] < c[0]:
                x01.append(point)
            else:
                x23.append(point)
        a, b = x01
        if a[1] > b[1]:
            x0, x1 = a, b
        else:
            x0, x1 = b, a
        a, b = x23
        if a[1] > b[1]:
            x3, x2 = a, b
        else:
            x3, x2 = b, a
        result = x0, x1, x2, x3
    return result


def get_polygon(contour):
    epsilon = 0.1 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon


def get_quadrilateral(contour):
    max_area = 0
    quadrilateral = None
    epochs = 2000
    contour = contour.reshape((-1, 2))
    for it in range(epochs):
        idxs = np.random.choice(range(len(contour)), size=4, replace=False)
        points = contour[idxs, :]
        area = cv2.contourArea(points)
        if area > max_area:
            max_area = area
            quadrilateral = points
    return quadrilateral


def get_warping(q, plate_shape):
    warp = None
    if q is not None:
        w, h = plate_shape
        p1 = [0, 0]
        p2 = [w - 1, 0]
        p3 = [w - 1, h - 1]
        p0 = [0, h - 1]
        dst = np.array([p1, p2, p3, p0], dtype=np.float32)
        x0, x1, x2, x3 = get_xs(q)
        q = np.array([x1, x2, x3, x0], dtype=np.float32)
        warp = cv2.getPerspectiveTransform(q, dst)
    return warp


def warp_image(im, warp, plate_shape):
    if warp is not None:
        im = cv2.warpPerspective(im, warp, plate_shape)
    return im


def rotate_image(image, center, theta):
    """
    Rotates image around center with angle theta in radians
    """

    theta_degrees = theta * 180 / np.pi
    shape = (image.shape[1], image.shape[0])
    center = tuple(center)
    matrix = cv2.getRotationMatrix2D(
        center=center, angle=theta_degrees, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)
    return image


def crop_image(im, rectangle):
    x0, x1, x2, x3 = rectangle
    w = np.linalg.norm(x3 - x0)
    h = np.linalg.norm(x1 - x0)
    cx, cy = x1
    dh = 5
    dw = 5
    im = im[max(0, cy - dh):min(im.shape[1], int(cy + h + dh)),
         max(0, cx - dw):min(im.shape[1], int(cx + w + dw))]
    return im


def get_theta(x0, x3):
    tan_theta = (x3[1] - x0[1]) / (x3[0] - x0[0])
    theta = np.arctan(tan_theta)
    return theta


def print_named_images(images, folder, name):
    print(f"Saving {name} images")
    for image_idx, im in enumerate(images):
        save_image(im, f"{folder}/{name}_{image_idx}.png")


def has_dark_font(im):
    h, w = im.shape
    low_t = 0.1
    high_t = 0.9
    im = im[int(low_t * h):int(high_t * h), int(low_t * w):int(high_t * w)]
    im = cv2.threshold(im, im.mean(), 255, cv2.THRESH_BINARY)[1]
    result = im.mean() > 255 / 2
    return result


def get_binary_im(im):
    if im is not None:
        im = cv2.threshold(im, im.mean(), 255, cv2.THRESH_BINARY)[1]
    return im


def get_center_point(r):
    if r is not None:
        r = r.mean(axis=0).astype(int)
    return r


def get_y_limits(im):
    borders = ~(im[:, :, 0].mean(axis=1) > 0.20 * 255)
    return borders


def print_limits(im):
    borders = ~(im.mean(axis=1) > 0.20 * 255)
    mp = int(len(im) / 2)
    up, lp = None, None
    for idx in range(mp, 0, -1):
        if borders[idx]:
            up = idx
            break
    for idx in range(mp, len(im), 1):
        if borders[idx]:
            lp = idx
            break
    if up is not None and lp is not None:
        im[0:up, :] = 0
        im[lp:len(im), :] = 0
    return im
