# -*- coding: utf-8 -*-
"""Run rolling shutter attack on object detection models.

This script overlays the rolling the shutter-introduced
pattern (distortion) on frames extracted from videos, and 
computes the effect of the pattern on an object detection 
model by comparing the output of the malicious frame with
the output of a legitimate (non-distorted) frame. Results
are saved in 

/home/data/results/object_detection/<video_name>/...

Example (run these two sequentially):
    Compute the normal performance (baseline) of the object
    detector on the frames extracted from video b1c9c847-3bda4659::

        $ python attack_object_detection.py \
            --pattern_filepath baseline \
            --video_name b1c9c847-3bda4659 \
            --model_name ssd_inception_v2_coco_2018_01_28

    Compute the corrupted performance (under attack) of the
    object detector on the frames extracted from video
    b1c9c847-3bda4659, and post-process the difference 
    between legitimate and under-attack performance::

        $ python attack_object_detection.py \
            --pattern_filepath /home/data/results/extracted_patterns/Axis/259Hz/Exposure\ 75/5/40.png \
            --video_name b1c9c847-3bda4659 \
            --model_name ssd_inception_v2_coco_2018_01_28

"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
import tensorflow as tf
import yaml
import skvideo.io
from object_detection.utils import label_map_util

from joblib import Parallel, delayed
import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_model(model_name: str) -> tf.keras.Model:
    """Load a model located in /home/data/models/<model_name>.

    Returns:
        A tensorflow model.

    """
    model_dir = "/home/data/models/{}/saved_model".format(model_name)
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures["serving_default"]
    return model


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


def extract_pattern_mask(patt_mask: np.array) -> np.array:
    """Flattens a 3d mask across the channel dimension.

    Args:
        patt_mask: pattern mask, np.array of shape [h,w,3]

    Returns:
        The flattened mask.
    """
    flat_p = patt_mask.reshape(-1, 3)  # (h * w, 3)
    flat_p_sum = flat_p.sum(axis=1)  # h*w
    mask_flat = flat_p_sum != 0
    mask = mask_flat.reshape(patt_mask.shape[0], patt_mask.shape[1])
    return mask


def print_some(df: pd.DataFrame) -> str:
    """Prints some information from a dataframe."""
    grpd = df[["type", "image_id"]].groupby("type").count()
    indx = grpd.index.values
    string = " - "
    for i in indx:
        string += "{}: {}, ".format(i, grpd.loc[i]["image_id"])
    string += "\n"
    return string


def setup_output_f(folder: str) -> bool:
    """Creates output folder for object detection results."""
    os.makedirs(f"{folder}/pickle/", exist_ok=True)
    os.makedirs(f"{folder}/results/", exist_ok=True)
    return True


def postprocess_attack_results(
    batch_i: int,
    baseline_dir: str,
    output_dir: str,
    imgs_fpaths: list,
    pattern_r: np.array,
    batch_size: int,
    imgh: int,
    imgw: int,
) -> pd.DataFrame:
    """Post-processing for object detection results.

    Args:
        batch_i: index of the image batch.
        baseline_dir: folder containing baseline results.
        output_dir: folder containing rsa-corrupted results.
        imgs_fpaths: list of images filepaths.
        pattern_r: np.array containing the pattern image.
        batch_size: batch size.
        imgh: height of images.
        imgw: width of images.

    Returns:
        A pandas dataframe containing the results.
    """
    # load pickled files where inference results are stored.
    org_fpath = f"{baseline_dir}/pickle/{str(batch_i).zfill(5)}.p"
    adv_fpath = f"{output_dir}/pickle/{str(batch_i).zfill(5)}.p"
    if not os.path.isfile(org_fpath) or not os.path.isfile(adv_fpath):
        print(f"baseline or corrupted pickle file {str(batch_i).zfill(5)}.p missing.")
        return None
    output_dicts_org = pickle.load(open(org_fpath, "rb"))
    output_dicts_adv = pickle.load(open(adv_fpath, "rb"))

    # load the corresponding images
    batch_start, batch_end = batch_i, int(batch_i + batch_size)
    imgs = load_imgs_batch(imgs_fpaths[batch_start:batch_end], target_size=(imgh, imgw))
    # apply the pattern
    advs = utils.apply_pattern(imgs, pattern_r)

    # compute attack effect
    result_df_list = [
        utils.compute_attack_effect(
            output_dicts_org[i],
            output_dicts_adv[i],
            imgs[i],
            advs[i],
            pattern_mask=extract_pattern_mask(pattern_r),
        )
        for i in range(len(imgs))
    ]

    # save parameters to dataframe
    for i in range(len(imgs)):
        ith_fname = imgs_fpaths[batch_start:batch_end][i].split(os.sep)[-1]
        result_df_list[i]["image_id"] = ith_fname
    result_df = pd.concat(result_df_list, ignore_index=True)
    result_df["batch_id"] = batch_i
    return result_df


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


def get_ds_iterator(img_fpaths: list, batch_size: int, height: int, width: int):
    """Creates an iterator (tf.data.Iterator) from a list of image filepaths.

    Args:
        img_fpaths: list of images filepaths.
        batch_size: size of batches.
        height: resize images to this height.
        width: resize images to this width.

    Returns:
        A tf.data.Iterator iterating over the loaded images.
    """
    dataset = tf.data.Dataset.from_tensor_slices(img_fpaths)
    dataset = dataset.map(
        lambda x: load_img(x, h=height, w=width), num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.__iter__()


def run_inference_for_batch(model: tf.keras.Model, imgs: list) -> dict:
    """Runs inference on images with model.

    Args:
        model: model to use for inference.
        imgs: list of arrays containing images in [h,w,3]

    Returns:
        A dictionary with the inference results.
    """
    imgs = np.asarray(imgs)
    batch_size = imgs.shape[0]

    assert imgs.ndim == 4
    input_tensor = tf.convert_to_tensor(imgs)
    output_dict = model(input_tensor)

    # do a few processing steps on outputs to bring them into
    # numpy arrays.
    num_detections = int(output_dict.pop("num_detections")[0])

    dict_outputs = []
    for i in range(batch_size):
        ith_out_dict = {
            key: value[i, :num_detections].numpy() for key, value in output_dict.items()
        }
        ith_out_dict["num_detections"] = num_detections
        ith_out_dict["detection_classes"] = ith_out_dict["detection_classes"].astype(
            np.int64
        )
        dict_outputs.append(ith_out_dict)

    return dict_outputs


def filter_detections(inference_dict: dict, score_min: float = 0.5) -> dict:
    """Removes non-confident detections from a dictionary of detections.

    Args:
        inference_dict: inference output dictionary.
        score_min: the cutoff threshold to use to filter detections out.

    Returns:
        A new dictionary with confident-detections (>score_min) only.
    """
    new_dict = {
        "detection_boxes": [],
        "detection_scores": [],
        "detection_classes": [],
        "num_detections": inference_dict["num_detections"],
    }

    for i in range(len(inference_dict["detection_scores"])):
        if inference_dict["detection_scores"][i] >= score_min:
            new_dict["detection_boxes"].append(
                inference_dict["detection_boxes"][i].tolist()
            )
            new_dict["detection_scores"].append(
                inference_dict["detection_scores"][i].tolist()
            )
            new_dict["detection_classes"].append(
                inference_dict["detection_classes"][i].tolist()
            )

    new_dict["detection_boxes"] = np.array(new_dict["detection_boxes"])
    new_dict["detection_scores"] = np.array(new_dict["detection_scores"])
    new_dict["detection_classes"] = np.array(new_dict["detection_classes"])
    return new_dict


def load_imgs_batch(imgs_filepaths: list, target_size: tuple) -> np.array:
    """Load a list of images into numpy arrays.

    Args:
        imgs_filepaths: Images absolute file paths.
        target_size: Tuple (w,h), images are reshaped to this dimension.

    Returns:
        A numpy array shaped [n,h,w,3] with the loaded images.
    """
    imgs = []
    for i, _ in enumerate(imgs_filepaths):
        img = skimage.io.imread(imgs_filepaths[i])
        if img.ndim == 2:  # if grayscale add channel axis
            img = np.tile(img[..., np.newaxis], (1, 1, 3))
        if target_size and len(target_size) == 2:
            img = skimage.transform.resize(
                img, target_size, preserve_range=True
            ).astype(np.uint8)
        imgs.append(img)
    return np.array(imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate attack against object_detection."
    )
    parser.add_argument("-f", "--pattern_filepath", type=str, required=True)
    parser.add_argument("-v", "--video_name", type=str, required=True)
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        type=str,
        choices=[
            "ssd_inception_v2_coco_2018_01_28",
            "faster_rcnn_inception_v2_coco_2018_01_28",
        ],
    )
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("--n_jobs", default=1, type=int)
    parser.add_argument(
        "-l",
        "--labels_path",
        default="/home/models/research/object_detection/data/mscoco_label_map.pbtxt",
        type=str,
    )

    args = parser.parse_args()
    params = yaml.load(open("/home/rsa/config.yaml"), Loader=yaml.FullLoader)
    args = merge_args(args, params["defaults"])
    video_id = args.video_name.split(".")[0]

    assert os.path.isfile(args.pattern_filepath) or args.pattern_filepath == "baseline"

    ds_args = params["datasets"][args.dataset_name]
    inp_h, inp_w = ds_args["network_input_height"], ds_args["network_input_width"]

    tf.gfile = tf.io.gfile
    category_index = label_map_util.create_category_index_from_labelmap(
        args.labels_path, use_display_name=True
    )

    model = load_model(args.model_name)

    if args.pattern_filepath != "baseline":
        # setup pattern
        pattern_fpath = os.path.abspath(args.pattern_filepath)
        # the absolute path looks like this
        # /home/data/results/extracted_patterns/Axis/25Hz/Exposure 75/5/<FRAMEno>.png
        _, _, _, _, _, cam, freq, exp, dc, filename = pattern_fpath.split(os.path.sep)
        pattern_id = filename.split(".")[0]
        pattern = skimage.io.imread(pattern_fpath)
        pattern_r = skimage.transform.resize(
            pattern, (inp_h, inp_w), preserve_range=True
        ).astype(np.uint8)
        pattern_mask = extract_pattern_mask(pattern_r)
        output_dir = f"/home/data/results/object_detection/{args.dataset_name}/{video_id}/{freq}/{exp}/{dc}/{pattern_id}/{args.model_name}"
        setup_output_f(output_dir)
        skimage.io.imsave(f"{output_dir}/pattern.png", pattern, check_contrast=False)
    else:
        output_dir = f"/home/data/results/object_detection/{args.dataset_name}/baseline/{video_id}/{args.model_name}"
        setup_output_f(output_dir)

    img_fpaths = get_bdd100k_imgs_filepaths(ds_args["dataset_subdir"], args.video_name)
    n = len(img_fpaths)
    img_fpaths = img_fpaths[: n // args.batch_size * args.batch_size]
    dataset_iter = get_ds_iterator(img_fpaths, args.batch_size, inp_h, inp_w)

    for j in range(0, len(img_fpaths), args.batch_size):
        #  main loop for inference only, saves pickle results
        batch_start, batch_end = j, int(j + args.batch_size)
        sys.stdout.write(
            "\rImages %d:%d / %d" % (batch_start, batch_end, len(img_fpaths))
        )
        sys.stdout.flush()
        imgs = next(dataset_iter).numpy().astype(np.uint8)
        if args.pattern_filepath != "baseline":
            imgs = utils.apply_pattern(imgs, pattern_r)
        output_dicts = run_inference_for_batch(model, imgs)

        output_dicts = [filter_detections(od) for od in output_dicts]
        pickle.dump(
            output_dicts, open(f"{output_dir}/pickle/{str(j).zfill(5)}.p", "wb")
        )

    if args.pattern_filepath != "baseline":
        # if evaluating a pattern (not baseline), then do some post-processing to compute attack effect
        baseline_dir = f"/home/data/results/object_detection/{args.dataset_name}/baseline/{video_id}/{args.model_name}"
        result_dir = f"{output_dir}/results"
        if not os.path.isdir(baseline_dir):
            print(f"[ERROR] - baseline directory ({baseline_dir}/results) missing.")
            exit()
        else:
            dfs = Parallel(n_jobs=args.n_jobs)(
                delayed(postprocess_attack_results)(
                    batch_index,
                    baseline_dir,
                    output_dir,
                    img_fpaths,
                    pattern_r.copy(),
                    args.batch_size,
                    inp_h,
                    inp_w,
                )
                for batch_index in range(0, len(img_fpaths), args.batch_size)
            )
        results_df = pd.concat(dfs, ignore_index=True)
        results_df["model_name"] = args.model_name
        results_df["pattern_id"] = pattern_id
        sys.stdout.write(print_some(results_df))
        results_df.to_csv(f"{output_dir}/results/result.csv")
