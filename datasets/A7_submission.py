import collections

import numpy as np
import torch
import time

def classify_and_detect(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)
    t1 = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    for i in range(N):
        img = images[i].reshape(64, 64, 3)
        results = model(img, size=64).xyxy[0].cpu().numpy()
        if results.shape[0] == 0:
            continue
        sorted_conf_result = sorted(results, key=lambda kv: kv[4])
        digits = set()
        picked_results = []
        for result in sorted_conf_result[::-1]:
            digit = result[5]
            if digit not in digits:
                digits.add(digit)
                picked_results.append(result)
            else:
                continue
        picked_results = sorted(picked_results, key = lambda kv: kv[4])
        # insert one in case there is only one result
        sorted_picked_results = [picked_results[0]] + picked_results
        top_2_result = sorted_picked_results[-2:]
        top_2_sorted_result = sorted(top_2_result, key=lambda kv: kv[5])
        pred_class[i] = [top_2_sorted_result[0][5], top_2_sorted_result[1][5]]
        pred_bboxes[i][0] = [int(top_2_sorted_result[0][1]), int(top_2_sorted_result[0][0]), int(top_2_sorted_result[0][3]),
                             int(top_2_sorted_result[0][2])]
        pred_bboxes[i][1] = [int(top_2_sorted_result[1][1]), int(top_2_sorted_result[1][0]), int(top_2_sorted_result[1][3]),
                             int(top_2_sorted_result[1][2])]
    print(time.time() - t1)
    return pred_class, pred_bboxes
