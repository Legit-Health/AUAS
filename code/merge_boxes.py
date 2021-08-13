import os

import numpy as np
import pandas as pd
import cv2

import scipy.ndimage as ndimage


def calculate_UAS(tags):
    counts = len(tags)
    if counts == 0:
        uas_score = 0
        uas_sev = "None"
    elif 0 < counts < 20:
        uas_score = 1
        uas_sev = "Mild"
    elif 20 <= counts < 50:
        uas_score = 2
        uas_sev = "Moderate"
    else:
        uas_score = 3
        uas_sev = "Severe"
    return uas_score, uas_sev, counts


def plot_bboxes(path_img, boxes, max_size, lesion_colours, thickness=None, show_img=True, show_class=False,
                show_person=False, title='Image', waitTime=0):
    img = cv2.imread(path_img, -1)
    img, ratio = smart_resize(img, max_size=max_size)
    thickness = thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    for idx, b in boxes.iterrows():
        label, person, x, y, w, h = list(b[['label', 'person', 'x', 'y', 'w', 'h']])
        x = int(round(x * ratio))
        y = int(round(y * ratio))
        w = int(round(w * ratio))
        h = int(round(h * ratio))
        cx = int(round(x + 0.5 * w))
        cy = int(round(y + 0.5 * h))
        img = cv2.rectangle(img, (x, y), (x + w, y + h), lesion_colours[person], thickness=thickness)
        img = cv2.circle(img, (cx, cy), 2, lesion_colours[person], thickness=1)
        label_txt = label if show_class else ""
        label_txt = label_txt + "({})".format(person) if show_person else label
        if show_class or show_person:
            img = cv2.putText(img, label_txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lesion_colours[person], 2)

    if show_img:
        cv2.imshow(title, img)
        cv2.waitKey(waitTime)

    return img


def smart_resize(img, max_size=500):
    ratio = max_size / max(list(img.shape))
    resized = cv2.resize(img, None, fx=ratio, fy=ratio)
    return resized, ratio


def mask_from_boxes(data, shape_tuple):
    img = np.zeros(shape_tuple)
    for idx, row in data.iterrows():
        img = cv2.rectangle(img, (int(row['x']), int(row['y'])),
                            (int(row['x']) + int(row['w']), int(row['y']) + int(row['h'])), (255, 255, 255),
                            thickness=-1)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255
    return img


def compute_doctor_scores(data, path_img):
    doctors = list(data['person'].unique())
    scores = {d: [] for d in sorted(doctors)}
    for d in doctors:
        print("Doctor ", d)
        data_d = data[data['person'] == d]
        other_docs = [o for o in doctors if o != d]
        imgs = list(data_d['img'].unique())
        for f in imgs:
            data_f = data_d[data_d['img'] == f]
            img = cv2.imread(os.path.join(path_img, f))
            img_d = mask_from_boxes(data_f, tuple(img.shape))
            s = []
            for d2 in other_docs:
                d2_data = data[(data['person'] == d2) & (data['img'] == f)]
                if len(d2_data) > 0:
                    img_d2 = mask_from_boxes(d2_data, tuple(img.shape))
                    dice = 2.0 * np.sum(img_d * img_d2) / np.sum(img_d + img_d2)
                    s.append(dice)
            s_mean = np.asarray(s).mean()
            scores[d].append(s_mean)
        scores[d] = np.asarray(scores[d]).mean()

    total = np.asarray(list(scores.values())).sum()
    for d in scores.keys():
        scores[d] = scores[d] / total

    return scores


def compute_doctor_local_scores(data, img):
    doctors = list(data['person'].unique())
    scores = {}
    for d in doctors:
        d_data = data[data['person'] == d]
        img_d = mask_from_boxes(d_data, tuple(img.shape))
        s = []
        other_docs = [o for o in doctors if o != d]
        for d2 in other_docs:
            d2_data = data[data['person'] == d2]
            img_d2 = mask_from_boxes(d2_data, tuple(img.shape))
            dice = 2.0 * np.sum(img_d * img_d2) / np.sum(img_d + img_d2)
            s.append(dice)
        s = np.asarray(s).mean()
        scores[d] = s

    score_total = np.asarray(list(scores.values())).sum()
    scores = {k: v / score_total for k, v in scores.items()}
    return scores


def generate_prob_map_weighted(img, data, score_weights=None, gaussian_A=3, sigma_pct_lower=0.25, sigma_pct_upper=0.70,
                               eps=0.005):
    map = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype='float')
    w_map = np.zeros_like(map)
    h_map = np.zeros_like(map)

    # If we don't specify overall score weights,
    # we will compute local annotation scores for each doctor
    if not score_weights:
        score_weights = compute_doctor_local_scores(data, img)
        print("Local scores: ", score_weights)

    for idx, row in data.iterrows():
        X, Y = np.meshgrid(range(0, map.shape[1]), range(0, map.shape[0]))
        center_x = int(row['x'] + row['w'] * 0.5)
        center_y = int(row['y'] + row['h'] * 0.5)
        size_ratio = row['w'] * row['h'] / (map.shape[0] * map.shape[1])
        # If a box is TOO small compared to the image, the Gaussian distribution will be too thin
        if size_ratio < 0.0005:
            # In those cases we will use a different "hat width"
            sigma_pct_i = sigma_pct_upper
        else:
            # Otherwise we use the same width
            sigma_pct_i = sigma_pct_lower
        sigma_x, sigma_y = sigma_pct_i * 0.5 * row['w'], sigma_pct_i * 0.5 * row['h']
        gauss_term_x = 0.5 * ((X - center_x) / sigma_x) ** 2
        gauss_term_y = 0.5 * ((Y - center_y) / sigma_y) ** 2
        gaussian_i = np.clip(gaussian_A * np.exp(-(gauss_term_x + gauss_term_y)), 0, 1)
        map += gaussian_i * score_weights[row['person']]  # Weighted sum

        w_map += (gaussian_i > eps).astype(float) * gaussian_i * row['w'] * score_weights[row['person']]
        h_map += (gaussian_i > eps).astype(float) * gaussian_i * row['h'] * score_weights[row['person']]

    map_norm = map / map.max()

    # Weighted average of bbox width and height
    map[map == 0] = 1
    w_map /= map
    h_map /= map

    return map_norm, w_map, h_map


def process_map(map_data, img, img_name, img_output_size=500, prob_thresh=0.25):
    map, w_map, h_map = map_data

    # Filtering and thresholding
    map_cleaned = map.copy()
    map_min = ndimage.minimum_filter(map, size=5, mode='constant')
    map_cleaned[map_min < prob_thresh] = 0
    centers = 255 * (map_cleaned > 0).astype(np.uint8)

    # For generating the result image
    img, ratio = smart_resize(img, max_size=img_output_size)
    img_result = img.copy()

    # Finding the blobs/contours in the binary image
    boxes = []
    contours, _ = cv2.findContours(centers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        M['m00'] = max(1, M['m00'])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        w = int(w_map[cY, cX])
        h = int(h_map[cY, cX])
        x = max(0, cX - int(w / 2))
        y = max(0, cY - int(h / 2))

        if w > 0 and h > 0:
            lt = round(0.002 * (img_result.shape[0] + img_result.shape[1]) / 2) + 1
            x_plot, y_plot = int(x * ratio), int(y * ratio)
            w_plot, h_plot = int(w * ratio), int(h * ratio)
            cX_plot, cY_plot = int(cX * ratio), int(cY * ratio)
            img_result = cv2.rectangle(img_result, (x_plot, y_plot), (x_plot + w_plot, y_plot + h_plot), (101, 175, 4),
                                       thickness=lt)
            img_result = cv2.circle(img_result, (cX_plot, cY_plot), 3, (45, 42, 42), -1)
            boxes.append([img_name, 'Hive', x, y, w, h])

    # Creating the output dataframe
    boxes_df = pd.DataFrame(boxes, columns=['img', 'label', 'x', 'y', 'w', 'h'])
    return boxes_df, img_result, [map, map_min, map_cleaned, centers]


if __name__ == "__main__":

    # BGR
    # Set as many colours as annotators you have
    lesion_colours = {0: (101, 175, 4),  # merged labels
                      1: (45, 42, 42),  # Person 1
                      2: (45, 42, 42),  # Person 2
                      3: (45, 42, 42),  # ...
                      4: (45, 42, 42),
                      5: (45, 42, 42)}

    path_dataset = "../data/"
    path_images = path_dataset + "images/"
    path_labels = path_dataset + "labels/"

    # Load raw annotation data
    data_table = pd.read_csv(os.path.join(path_labels, "labels.csv"))

    # Compute the doctor scores globally (dataset level)
    if os.path.isfile(os.path.join(path_labels, "doctor_scores.csv")):
        scores = pd.read_csv(os.path.join(path_labels, "doctor_scores.csv")).loc[0].to_dict()
        scores = {int(k): v for k, v in scores.items()}  # Make sure the keys are integers
    else:
        scores = compute_doctor_scores(data_table, path_images)
        print("Global scores: ", scores)

    files = list(data_table['img'].unique())
    final_boxes = pd.DataFrame(columns=['img', 'label', 'x', 'y', 'w', 'h'])
    for i, f in enumerate(files):
        print("({}) - {}".format(i, f))
        boxes = data_table[data_table['img'] == f][['label', 'person', 'x', 'y', 'w', 'h']].reset_index(drop=True)
        # Image with original labels
        img_pre = plot_bboxes(os.path.join(path_images, f), boxes, 500, lesion_colours,
                              title='Original labels',
                              show_person=False)

        # Merging labels
        img = cv2.imread(os.path.join(path_images, f))
        map_data = generate_prob_map_weighted(img, boxes, scores,
                                              sigma_pct_lower=0.30,
                                              sigma_pct_upper=0.80)

        p_thresh = 2 / len(scores.keys())  # At least 2/N annotators
        boxes_processed, img_post, map_data_processed = process_map(map_data, img, f,
                                                                    img_output_size=500,
                                                                    prob_thresh=p_thresh)
        final_boxes = pd.concat([final_boxes, boxes_processed])

        cv2.imshow("Processed labels", img_post)
        cv2.waitKey()

    final_boxes.to_csv(os.path.join(path_labels, "labels_merged.csv"), index=False)
