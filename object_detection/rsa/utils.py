import skimage.io
import skvideo.io
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


def apply_pattern(base: np.array, pattern: np.array) -> np.array:
    """Overlays a pattern on a base image.

    Args:
        base: array containing the image(s) to overlay the pattern on.
              Shaped [h,w,3] or [n,h,w,3] if overlaying the pattern on
              a set of images.
        pattern: array containing the rolling shutter pattern.
              Shaped [h,w,3] or [n,h,w,3] if overlaying multiple patterns
              on multiple images.

    Returns:
        An image where pattern is overlayed onto base.
    """
    assert base.dtype == np.uint8
    assert pattern.dtype == np.uint8
    if base.ndim == 4:
        assert base.shape[1:] == pattern.shape[1:] or base.shape[1:] == pattern.shape
    elif base.ndim == 3:
        assert base.shape == pattern.shape
    else:
        raise Exception(
            "Incorrect number of dimensions for 'base' in 'apply_pattern', 3 or 4 expected"
        )
    b = base.astype(float)
    p = pattern.astype(float)
    r = np.clip(b + p, 0, 255).astype(np.uint8)
    return r


def compute_attack_effect(
    dict_org: dict,
    dict_adv: dict,
    image_ori: np.array,
    image_adv: np.array,
    pattern_mask: np.array,
) -> pd.DataFrame:
    """Compute the effect of the RSA corrupting on the inference outputs.

    Args:
        dict_org: a dictionary containing the legitimate inference results.
        dict_adv: a dictionary containing the (rsa-)corrupted inference results.
        image_ori: original image.
        image_adv: corrupted image.
        pattern_mask: 2-d numpy array containing a pattern mask [h,w]

    Returns:
        A pandas dataframe containing the results.
    """

    assert image_ori.shape == image_adv.shape

    columns = ["type", "org_y1", "org_x1", "org_y2", "org_x2", "org_score", "org_class"]
    columns += ["adv_y1", "adv_x1", "adv_y2", "adv_x2", "adv_score", "adv_class"]
    columns += ["iou", "overlap"]

    df1 = find_disappeared_or_misplaced(dict_org, dict_adv, pattern_mask)
    df2 = find_appeared(dict_org, dict_adv, pattern_mask)

    if len(df2) > 0:
        df1.extend(df2)

    return pd.DataFrame(df1, columns=columns)


def find_disappeared_or_misplaced(
    dict_org: dict,
    dict_adv: dict,
    pattern_mask: np.array,
) -> list:
    """Compare two inferences outputs to find disappeared or misplaced boxes.

    See how this function determines the outcome of boxes in paper Section 7.1.

    Args:
        dict_org: a dictionary containing the legitimate inference results.
        dict_adv: a dictionary containing the (rsa-)corrupted inference results.
        pattern_mask: 2-d numpy array containing a pattern mask [h,w]

    Returns:
        A list of lists with the results.
    """
    df = []
    for i, _ in enumerate(dict_org["detection_boxes"]):
        out_row = []
        o_box, o_score, o_class = (
            dict_org["detection_boxes"][i],
            dict_org["detection_scores"][i],
            dict_org["detection_classes"][i],
        )

        # overlap
        bm, im = get_overlap_with_pattern(o_box, pattern_mask)
        ol_ = im.astype(int).sum() / bm.astype(int).sum()

        # now greedily search for the box with larges iou in the adv_dict
        ious = np.zeros(len(dict_adv["detection_boxes"]))
        for j, _ in enumerate(dict_adv["detection_boxes"]):
            ious[j] = get_iou(o_box, dict_adv["detection_boxes"][j])

        if np.all(ious < 0.5):
            # there are no boxes in adv with ious >= .5
            # this is a successful undetection
            # now check the overlap between the pattern and the original object box which was undetected
            out_row.extend(
                ["hidden"]
                + o_box.tolist()
                + [o_score]
                + [o_class]
                + np.repeat(np.nan, 7).tolist()
            )

        else:
            # if there are adv_boxes with ious >= .5 then
            # if one of adv_boxes has the same class, that means that the org_box is just slightly misplaced
            # if none of adv_boxes has the same class, then the org_box is hidden

            best_matching_box_index = np.argmax(ious)
            best_matching_box = dict_adv["detection_boxes"][best_matching_box_index]
            best_matching_box_score = dict_adv["detection_scores"][
                best_matching_box_index
            ]
            best_matching_box_class = dict_adv["detection_classes"][
                best_matching_box_index
            ]
            best_matching_box_iou = ious[best_matching_box_index]

            if best_matching_box_class == o_class:
                if best_matching_box_iou < 0.95:
                    # print("Object of class {} Misplaced, iou {:.3f}".format(o_class, best_matching_box_iou))
                    out_row.extend(
                        ["misplaced"]
                        + o_box.tolist()
                        + [o_score]
                        + [o_class]
                        + best_matching_box.tolist()
                        + [
                            best_matching_box_score,
                            best_matching_box_class,
                            best_matching_box_iou,
                        ]
                    )
                else:
                    out_row.extend(
                        ["unaltered"]
                        + o_box.tolist()
                        + [o_score]
                        + [o_class]
                        + best_matching_box.tolist()
                        + [
                            best_matching_box_score,
                            best_matching_box_class,
                            best_matching_box_iou,
                        ]
                    )
            else:
                out_row.extend(
                    ["hidden"]
                    + o_box.tolist()
                    + [o_score]
                    + [o_class]
                    + np.repeat(np.nan, 7).tolist()
                )
        out_row.extend([ol_])
        df.append(out_row)
    return df


def get_overlap_with_pattern(box: np.array, pattern_mask: np.array) -> tuple:
    """Return a binary mask marking pixels of overlap between box and pattern_mask

    Args:
        box: array of 4 elements indicating the box location in y1,x1,y2,x2
        pattern_mask: 2-d numpy array containing a pattern mask [h,w]

    Returns:
        A box mask and the intersection box-pattern mask.
    """
    box_mask = np.zeros(pattern_mask.shape, dtype=np.bool)  # create box mask
    h, w = box_mask.shape
    int_box = [int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)]
    # mark pixels in the box location as True
    box_mask[int_box[0] : int_box[2], int_box[1] : int_box[3]] = True
    box_mask_flat = box_mask.flatten()
    # compute intersection mask
    intersection = np.bitwise_and(box_mask_flat, pattern_mask.reshape(-1))
    return box_mask, intersection.reshape(pattern_mask.shape)


def find_appeared(
    dict_org: dict,
    dict_adv: dict,
    pattern_mask: np.array,
) -> list:
    """Compare two inferences outputs to find appeared boxes.

    See how this function determines the outcome of boxes in paper Section 7.1.

    Args:
        dict_org: a dictionary containing the legitimate inference results.
        dict_adv: a dictionary containing the (rsa-)corrupted inference results.
        pattern_mask: 2-d numpy array containing a pattern mask [h,w]

    Returns:
        A list of lists with the results.
    """
    df = []
    # now check if any new object has appeared in adv
    for i, _ in enumerate(dict_adv["detection_boxes"]):
        out_row = []
        a_box, a_score, a_class = (
            dict_adv["detection_boxes"][i],
            dict_adv["detection_scores"][i],
            dict_adv["detection_classes"][i],
        )
        bm, im = get_overlap_with_pattern(a_box, pattern_mask)
        ol_ = im.astype(int).sum() / bm.astype(int).sum()
        # now greedily search for the box with larges iou in the adv_dict
        ious = np.zeros(len(dict_org["detection_boxes"]))
        for j, _ in enumerate(dict_org["detection_boxes"]):
            ious[j] = get_iou(a_box, dict_org["detection_boxes"][j])

        if np.all(ious < 0.5):
            out_row.extend(
                ["appeared"]
                + np.repeat(np.nan, 6).tolist()
                + a_box.tolist()
                + [a_score]
                + [a_class]
                + [np.nan, ol_]
            )
            df.append(out_row)
    return df


def get_iou(bb1: dict, bb2: dict) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bb1: dictionary with following keys {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2: dictionary with following keys {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

    Returns:
        The IoU between the two bounding boxes (float in [0,1])
    """

    bb1 = {
        "x1": bb1[1],
        "y1": bb1[0],
        "x2": bb1[3],
        "y2": bb1[2],
    }
    bb2 = {
        "x1": bb2[1],
        "y1": bb2[0],
        "x2": bb2[3],
        "y2": bb2[2],
    }

    assert bb1["x1"] <= bb1["x2"]
    assert bb1["y1"] <= bb1["y2"]
    assert bb2["x1"] <= bb2["x2"]
    assert bb2["y1"] <= bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"] + 1) * (bb1["y2"] - bb1["y1"] + 1)
    bb2_area = (bb2["x2"] - bb2["x1"] + 1) * (bb2["y2"] - bb2["y1"] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def merge_args(
    argparse_namespace: argparse.Namespace, yaml_dict: dict
) -> argparse.Namespace:
    """Merge arguments in a namespace with those in a dictionary.

    Overrides each None argument in the namespace if the same key
    is defined in the dictionary, adds arguments that are in the
    dictionary to the namespace.

    Args:
        argparse_namespace: argument namespace.
        yaml_dict: argument dictionary.

    Returns:
        A namespace populated with the arguments from the dictionary.
    """
    args_dict = vars(argparse_namespace)
    overwritten = []
    for key in args_dict.keys():
        # overwrite parameters
        if args_dict[key] is None and key in yaml_dict:
            args_dict[key] = yaml_dict[key]
            overwritten.append(key)
    # add additional default parameters
    for key in yaml_dict:
        if key not in overwritten:
            args_dict[key] = yaml_dict[key]
    return argparse.Namespace(**args_dict)


def get_bdd100k_imgs_filepaths(
    subdir: str, video_name: str, frameskip: int = 10
) -> list:
    """Extracts frames from a video and returns the frames absolute filepaths.

    Args:
        subdir: subfolder in /home/data/datasets/bdd100k/videos/ where to
                look for the video and extract frames.
        video_name: filename of video file in
                    /home/data/datasets/bdd100k/videos/<subdir>
        frameskip: extract one frame every <frameskip> frames

    Returns:
        A list of absolute filepaths pointing to the extracted frames.
    """
    basedir = "/home/data/datasets/bdd100k/videos/"
    # where to find the individual frames
    single_frames_folder = f'{basedir}/{subdir}/{video_name.split(".")[0]}'
    video_path = f"{basedir}/{subdir}/{video_name}"
    # if frames where never extracted, then this function
    # extracts them.
    if (
        not os.path.isdir(single_frames_folder)
        or len(os.listdir(single_frames_folder)) < 500 / frameskip
    ):
        # then create a folder which will hold the extracted frames
        os.makedirs(single_frames_folder, exist_ok=True)
        frames = skvideo.io.vreader(video_path)
        print("Extracting frames for {}".format(video_name))
        for i, frame in enumerate(frames):
            if i % frameskip == 0:
                out_fpath = f"{single_frames_folder}/{str(i).zfill(5)}.jpg"
                skimage.io.imsave(out_fpath, frame, check_contrast=False)

    all_fnames = list(
        filter(lambda x: x[-4:] == ".jpg", os.listdir(single_frames_folder))
    )
    all_fpaths = list(
        sorted(map(lambda x: os.path.join(single_frames_folder, x), all_fnames))
    )
    return all_fpaths


def load_model(model_name: str) -> tf.keras.Model:
    """Load a model located in /home/data/models/<model_name>.

    Returns:
        A tensorflow model.

    """
    model_dir = "/home/data/models/{}/saved_model".format(model_name)
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures["serving_default"]
    return model


def load_img(file_path: str, height: int, width: int) -> tf.Tensor:
    """Loads tensorflow Tensors from jpg images filepaths.

    Args:
        file_path: image filepath.
        height: resize image to this height
        width: resize image to this width

    Returns:
        A tf.Tensor with the resized image.
    """
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width], method="bilinear")
    return img