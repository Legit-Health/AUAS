from utils import *


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
