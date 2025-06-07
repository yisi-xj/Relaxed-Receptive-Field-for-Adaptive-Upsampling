import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LABEL_NAMES = CLASSES = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag']


def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    new_x = cv2.resize(x, dsize=(w,h), interpolation=cv2.INTER_CUBIC)

    return new_x



def hist(pd, gt, nclass):
    def fast_hist(a, b, n):
        # a for gt
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n**2).reshape(n, n)
    return fast_hist(gt.flatten(), pd.flatten(), nclass)


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param dilation_ratio: (numpy array, uint8): binary mask
    :param mask: (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def calc_biou(pred, gt, num_classes, dilation_ratio=0.005):
    BOUNDARY_ID = num_classes + 1
    pred_boundary = pred.copy()
    gt_boundary = gt.copy()
    for i in range(num_classes):
        binary_mask = (pred_boundary == i).astype('uint8')
        binary_boundary = mask_to_boundary(binary_mask, dilation_ratio)
        pred_boundary[binary_mask > 0] = BOUNDARY_ID
        pred_boundary[binary_boundary > 0] = i
        binary_mask = (gt_boundary == i).astype('uint8')
        binary_boundary = mask_to_boundary(binary_mask, dilation_ratio)
        gt_boundary[binary_mask > 0] = BOUNDARY_ID
        gt_boundary[binary_boundary > 0] = i
    idx = pred_boundary < num_classes
    pd = pred_boundary[idx]
    gt = gt_boundary[idx]
    return hist(pd, gt, num_classes)


def compute_iou(cm):
    # iou = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))
    # miou = np.nanmean(iou)
    iou = np.zeros(cm.shape[0])
    num = np.diag(cm)
    den = cm.sum(1) + cm.sum(0) - np.diag(cm)
    idx = den > 0
    iou[idx] = num[idx] / den[idx]
    miou = np.mean(iou)
    return iou, miou


def eval_boundary_iou(predictions, ground_truths, num_classes=150):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    print('num_classes:', num_classes)
    print('ground_truths:', ground_truths)
    print('predictions:', predictions)
    for oup, gt in tqdm(zip(predictions, ground_truths)):
        cm += calc_biou(oup, gt, num_classes)
    iou, miou = compute_iou(cm)
    label_names = LABEL_NAMES
    ious = '; '.join(
        map(
            lambda x: '{}:{:.4f}'.format(x[0], x[1]),
            zip(label_names, iou)
        )
    )
    print(
        'BoundaryIoUs: [{0}], mBoundaryIoU: {1:.4f}'
            .format(ious, miou)
    )