import tensorflow.compat.v1 as tf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.utils import shuffle
import json
import time
import yaml
import utils
from PIL import Image, ImageDraw

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE


def apply_pattern_tf(
    image: tf.Tensor, pattern: tf.Tensor, apply: float = 0.0, pattern_fpath: str = None
) -> tuple:
    """Overlays a pattern onto an image tensor of the same shape.

    Args:
        image: image tensor.
        pattern: pattern tensor.
        apply: whether to apply the pattern (1) or to not apply it (0)
        pattern_fpath: the filepath of the pattern (this is returned for convenience)

    Returns:
        A tuple with the applied image, the apply parameter and the filepath parameter
    """
    image = tf.clip_by_value(image + (pattern * tf.cast(apply, tf.float32)), 0, 255)
    return image, apply, pattern_fpath


def get_ds_iterator_with_pattern_and_labels(
    img_fpaths: list, batch_size: int, h: int, w: int, patterns_fpaths: list
) -> tf.data.Dataset:
    """Get a dataset iterator over images with potentially applied patterns on top.

    Args:
        img_fpaths: list of filepaths for the images to load.
        batch_size: batch size.
        h: resize loaded images to this height.
        w: resize loaded images to this width.
        patterns_fpaths: list of filepaths for the pattern images to overlay.

    Returns:
        An iterator over a tensorflow dataset.
    """

    n_images = len(img_fpaths)

    # create image dataset with images in (h, w)
    img_dataset = tf.data.Dataset.from_tensor_slices(img_fpaths)
    _lambda_load = lambda x: utils.load_img(x, height=h, width=w)
    img_dataset = img_dataset.map(_lambda_load, num_parallel_calls=AUTOTUNE)
    # choose which images to perturb with pattern (1 perturbs, 0 does not perturb)
    chosen_rsa_labels = np.round(np.random.rand(n_images)).astype(np.float64)

    # choose which patterns to perturb images with
    chosen_patterns_fpaths = np.random.choice(
        patterns_fpaths, size=len(img_fpaths), replace=True
    )
    # create pattern dataset with filepaths
    patterns_filepaths = tf.data.Dataset.from_tensor_slices(chosen_patterns_fpaths)
    # load chosen patterns
    pattern_dataset = patterns_filepaths.map(_lambda_load, num_parallel_calls=AUTOTUNE)
    # load labels
    rsa_labels_dataset = tf.data.Dataset.from_tensor_slices(chosen_rsa_labels)

    # zip them together
    dataset = tf.data.Dataset.zip(
        (img_dataset, pattern_dataset, rsa_labels_dataset, patterns_filepaths)
    )

    # apply the function that merges the pattern onto the image based on the value of label
    _lambda_load2 = lambda img, pattern, label, pattern_fpath: apply_pattern_tf(
        img, pattern, label, pattern_fpath
    )
    dataset = dataset.map(_lambda_load2, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def get_test_iterator(
    img_fpaths_test: list,
    batch_size: int,
    h: int,
    w: int,
    chosen_pattern_test_filepaths: list,
    repeat: int = 1,
) -> tuple:
    """Get a dataset iterator over images with potentially applied patterns on top.

    Args:
        img_fpaths_test: list of filepaths for the images to load.
        batch_size: batch size.
        h: resize loaded images to this height.
        w: resize loaded images to this width.
        chosen_pattern_test_filepaths: list of filepaths for the pattern images to overlay.
        repeat: how many times to repeat the iteration over the dataset.

    Returns:
        Pointer to the get_next operation of the iterator.
    """
    ds_test = get_ds_iterator_with_pattern_and_labels(
        img_fpaths_test, batch_size, h, w, chosen_pattern_test_filepaths
    )
    ds_test = ds_test.repeat(repeat)
    iterator_test = tf.compat.v1.data.make_one_shot_iterator(ds_test)
    next_element_test = iterator_test.get_next()
    return next_element_test


def check_execution_time(
    sess: tf.Session, tensors_names: list, next_element, no_of_imgs: int
) -> list:
    """Checks the execution time of the tensorflow graph evaluating certain tensors.

    Args:
        sess: the tensorflow session.
        tensors_names: list of tensors to evaluate.
        next_element: pointer to the get_next operation of an iterator to read the data.
        no_of_imgs: number of images to forward to the network.

    Returns:
        A list containing the execution times for the images.
    """
    tensors_to_eval = [
        tf.get_default_graph().get_tensor_by_name(x) for x in tensors_names
    ]
    times = []
    for x in tqdm(range(no_of_imgs)):
        images_, _, _ = sess.run(next_element)
        start = time.time()
        _ = sess.run(tensors_to_eval, feed_dict={image_tensor: images_})
        end = time.time()
        times.append(end - start)
    return times


def add_shutter_head(
    feature_map: tf.Tensor,
    fully_conv: bool = False,
    n_neurons_before_dense: int = 0,
    kernel_size: list = None,
    strides: list = None,
    n_filters: int = None,
) -> tuple:
    """Adds an head to the network to do rolling shutter attack detection

    Args:
        feature_map: tensor containing the feature map to use as input.
        fully_conv: whether to use GlobalPooling or to flatten the computed feature map before logits layer.
        n_neurons_before_dense: number of neurons in the last layer before the dense layer (used for reshaping).
        kernel_size: convolution kernel.
        strides: convolution strides.
        n_filters: convolution number of filters.

    Returns:
        A tuple of tensor pointers.
    """
    with tf.variable_scope("shutter_head"):

        conv1 = tf.compat.v1.layers.Conv2D(
            n_filters, kernel_size, strides=strides, padding="valid", name="conv1"
        )(feature_map)
        conv1 = tf.nn.leaky_relu(conv1, name="conv1_activations")
        conv1 = tf.compat.v1.layers.MaxPooling2D(
            (2, 2), strides=(2, 2), padding="valid"
        )(conv1)
        conv1 = tf.compat.v1.layers.dropout(conv1)
        last = tf.identity(conv1, name="conv1_dropout")

        if fully_conv:
            pooled = tf.keras.layers.GlobalMaxPool2D()(last)
            dense1 = tf.compat.v1.layers.dense(
                pooled, n_filters, name="dense", kernel_regularizer="l2"
            )
        else:
            flat_fm = tf.keras.layers.Reshape(target_shape=(n_neurons_before_dense,))(
                last
            )
            dense1 = tf.compat.v1.layers.dense(
                flat_fm, n_filters, name="dense", kernel_regularizer="l2"
            )

        dense1 = tf.nn.leaky_relu(dense1, name="dense_activations")
        dense1 = tf.compat.v1.layers.dropout(dense1, name="dense_dropout")

        last_before_logits = tf.identity(dense1, name="last_before_logits")
        logits = tf.compat.v1.layers.dense(last_before_logits, 2, name="logits")

        logits = tf.identity(logits, name="logits")
        probs = tf.nn.softmax(logits, name="probs")
        shutter_labels = tf.placeholder(
            tf.float32, shape=(None, 2), name="shutter_labels"
        )
        accuracy = tf.reduce_mean(
            tf.keras.metrics.categorical_accuracy(shutter_labels, probs)
        )
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=shutter_labels
            )
        )
    return shutter_labels, logits, probs, accuracy, loss


def add_shutter_training(loss: tf.Tensor) -> tuple:
    """Adds an optimizer minimizing the given loss

    Args:
        loss: tensor pointing to the loss to minimize.

    Returns:
        A tuple of tensor pointers.
    """
    trainables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shutter_head")
    with tf.variable_scope("shutter_head"):
        with tf.variable_scope("training"):
            opt = tf.train.AdamOptimizer(learning_rate=1e-4)
            gd_step = opt.minimize(loss, var_list=trainables)
    return gd_step, opt, loss, trainables


def make_training_writer():
    """Adds necessary instructions for logging training information.

    Returns:
        A tuple of tensor pointers.
    """
    with tf.name_scope("performance"):
        # Summaries need to be displayed
        loss_ph_train = tf.placeholder(tf.float32, shape=None, name="loss_train_ph")
        loss_summary_train = tf.summary.scalar("loss_train", loss_ph_train)
        accuracy_ph_train = tf.placeholder(
            tf.float32, shape=None, name="accuracy_train_ph"
        )
        accuracy_summary_train = tf.summary.scalar("accuracy_train", accuracy_ph_train)

        loss_ph_val = tf.placeholder(tf.float32, shape=None, name="loss_val_ph")
        loss_summary_val = tf.summary.scalar("loss_val", loss_ph_val)
        accuracy_ph_val = tf.placeholder(tf.float32, shape=None, name="accuracy_val_ph")
        accuracy_summary_val = tf.summary.scalar("accuracy_val", accuracy_ph_val)

        loss_ph_test = tf.placeholder(tf.float32, shape=None, name="loss_test_ph")
        loss_summary_test = tf.summary.scalar("loss_test", loss_ph_test)
        accuracy_ph_test = tf.placeholder(
            tf.float32, shape=None, name="accuracy_test_ph"
        )
        accuracy_summary_test = tf.summary.scalar("accuracy_test", accuracy_ph_test)

        # add summary for input
        input_ph = tf.placeholder(
            tf.float32, shape=(None, None, None, 3), name="input_summary"
        )
        input_summary = tf.summary.image("input_summary", input_ph, max_outputs=5)

        # add summary for misclassifications
        mc_img_ph = tf.placeholder(
            tf.float32, shape=(None, None, None, 3), name="misclassif_img_ph"
        )
        mc_img_summ = tf.summary.image(
            "misclassif_img_summary", mc_img_ph, max_outputs=1000
        )
        mc_pfpath_ph = tf.placeholder(
            tf.string, shape=(None,), name="misclassif_pattern_filepath_ph"
        )
        mc_pfpath_summ = tf.summary.text(
            "misclassif_pattern_filepath_summary", mc_pfpath_ph
        )

        # Merge training and validation summaries together
        summ_train = tf.summary.merge([loss_summary_train, accuracy_summary_train])
        summ_val = tf.summary.merge([loss_summary_val, accuracy_summary_val])
        summ_test = tf.summary.merge(
            [loss_summary_test, accuracy_summary_test, mc_img_summ, mc_pfpath_summ]
        )

    return (
        summ_train,
        summ_val,
        summ_test,
        input_summary,
        loss_ph_train,
        accuracy_ph_train,
        loss_ph_val,
        accuracy_ph_val,
        loss_ph_test,
        accuracy_ph_test,
        input_ph,
        mc_img_ph,
        mc_pfpath_ph,
    )


def get_all_pattern_fpaths(
    folder: str = "/home/data/results/extracted_patterns/Axis/",
) -> list:
    """Exaustively walks through a directory and returns all found .png images absolute paths.

    Args:
        folder: folder to search.

    Returns:
        A list with the filepaths.
    """
    l = list(os.walk(folder))
    pattern_fpaths = []
    for f, sf, filenames in l:
        if len(filenames) > 0:
            pattern_fpaths.extend([os.path.join(f, fn) for fn in filenames])
    pattern_fpaths = set(filter(lambda x: x.endswith(".png"), pattern_fpaths))
    return pattern_fpaths


def get_train_val_test_split(
    n_videos_train: int,
    n_videos_val: int,
    n_videos_test: int,
    frameskip: int,
    random_seed: int = 42,
) -> tuple:
    """Splits the videos in the bdd100k folder into training-,validation- and test-set.

    Args:
        n_videos_train: number of videos to use for training.
        n_videos_val: number of videos to use for validation.
        n_videos_test: number of videos to use for test.
        frameskip: consider video frames every frameskip, skip the rest.
        random_seed:

    Returns:
        Three lists with the filepaths.
    """
    videos_list = list(
        filter(
            lambda x: x.endswith(".mov"),
            os.listdir("/home/data/datasets/bdd100k/videos/val"),
        )
    )
    videos_list = shuffle(videos_list, random_state=random_seed)

    assert len(videos_list) >= (n_videos_train + n_videos_val + n_videos_test)

    chosen_videos_train = videos_list[:n_videos_train]
    chosen_videos_val = videos_list[n_videos_train : n_videos_train + n_videos_val]
    chosen_videos_test = videos_list[
        n_videos_train + n_videos_val : n_videos_train + n_videos_val + n_videos_test
    ]
    fp_train, fp_val, fp_test = [], [], []
    for subset, filepaths in zip(
        [chosen_videos_train, chosen_videos_val, chosen_videos_test],
        [fp_train, fp_val, fp_test],
    ):
        for video_filename in subset:
            this_video_filepaths = utils.get_bdd100k_imgs_filepaths(
                "val", video_filename, frameskip=frameskip
            )
            filepaths.extend(this_video_filepaths)
    return fp_train, fp_val, fp_test


def evaluate_validation(sess: tf.Session, img_paths: list, next_element) -> tuple:
    """Evaluation on validation set.

    Args:
        sess: tensorflow session.
        img_paths: filepaths of images to use for validation.
        next_element: pointer to get_next element of validation set iterator.

    Returns:
        validation losses, accuracies and images.
    """
    losses, accuracies = [], []
    for j in tqdm(range(0, len(img_paths), args.batch_size)):
        images, labels, _ = sess.run(next_element)
        loss_, accuracy_ = sess.run(
            [loss, accuracy],
            feed_dict={
                image_tensor: images,
                shutter_labels: pd.get_dummies(labels).values,
            },
        )
        losses.append(loss_)
        accuracies.append(accuracy_)
    return np.array(losses), np.array(accuracies), images


def evaluate_test(sess: tf.Session, img_paths: list, next_element) -> tuple:
    """Evaluation on test set.

    Args:
        sess: tensorflow session.
        img_paths: filepaths of images to use for testing.
        next_element: pointer to get_next element of testing set iterator.

    Returns:
        test losses, accuracies, misclassifications and images.
    """
    p = tf.get_default_graph().get_tensor_by_name("shutter_head/probs:0")
    misclassif_images, losses, accuracies, misclassif_patterns = [], [], [], []

    truth = np.zeros(shape=(len(img_paths), 2))
    predicted_probs = np.zeros(shape=(len(img_paths), 2))
    pattern_fnames = []
    image_fnames = []

    for j in tqdm(range(0, len(img_paths), args.batch_size)):
        images_, labels_, pattern_fpaths_ = sess.run(next_element)
        one_hot_labels_ = pd.get_dummies(labels_).values
        loss_, accuracy_, p_ = sess.run(
            [loss, accuracy, p],
            feed_dict={image_tensor: images_, shutter_labels: one_hot_labels_},
        )
        losses.append(loss_)
        accuracies.append(accuracy_)

        # return these data
        jj = j // args.batch_size

        predicted_probs[jj * args.batch_size : (jj + 1) * args.batch_size] = p_
        truth[jj * args.batch_size : (jj + 1) * args.batch_size] = one_hot_labels_
        pattern_fnames.extend([x.decode() for x in pattern_fpaths_])
        image_fnames.extend(img_paths)

        predicted_labels_ = np.where(p_ > 0.5, 1, 0)
        for i in range(len(predicted_labels_)):
            if predicted_labels_[i, 0] != one_hot_labels_[i, 0]:
                # then it is mislassified
                PIL_img = Image.fromarray(np.uint8(images_[i]))
                drawer = ImageDraw.Draw(PIL_img)
                drawer.text((550, 350), f"p={p_[i][1]:.3f}", (0, 255, 0))
                misclassif_images.append(np.asarray(PIL_img).astype(np.float32))
                misclassif_patterns.append(pattern_fpaths_[i])
    result = {
        "probs": predicted_probs.tolist(),
        "ground_truth": truth.tolist(),
        "pattern_filepaths": pattern_fnames,
        "img_filepaths": img_paths,
    }
    return (
        np.array(losses),
        np.array(accuracies),
        misclassif_images,
        misclassif_patterns,
        result,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train new network head that detects images under Rolling Shutter Attack."
    )
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        choices=[
            "ssd_inception_v2_coco_2018_01_28",
            "faster_rcnn_inception_v2_coco_2018_01_28",
            "ssd_mobilenet_v2_coco_2018_03_29",
            "faster_rcnn_resnet50_coco_2018_01_28",
        ],
        type=str,
    )
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--input_image_width", type=int)
    parser.add_argument("--input_image_height", type=int)
    parser.add_argument("--num_videos_train", type=int)
    parser.add_argument("--num_videos_val", type=int)
    parser.add_argument("--num_videos_test", type=int)
    parser.add_argument("--ratio_train_patterns", type=int)
    parser.add_argument("--ratio_val_patterns", type=int)
    parser.add_argument("--ratio_test_patterns", type=int)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--validate_every", type=int)

    args = parser.parse_args()

    params = yaml.load(open("/home/rsa/config.yaml"), Loader=yaml.FullLoader)
    args = utils.merge_args(args, params["defense"]["defaults"])

    # get base images split
    img_fpaths_train, img_fpaths_val, img_fpaths_test = get_train_val_test_split(
        n_videos_train=args.num_videos_train,
        n_videos_val=args.num_videos_val,
        n_videos_test=args.num_videos_test,
        frameskip=10,
        random_seed=args.random_seed,
    )
    img_fpaths_train = img_fpaths_train[
        : len(img_fpaths_train) // args.batch_size * args.batch_size
    ]
    img_fpaths_val = img_fpaths_val[
        : len(img_fpaths_val) // args.batch_size * args.batch_size
    ]
    img_fpaths_test = img_fpaths_test[
        : len(img_fpaths_test) // args.batch_size * args.batch_size
    ]

    # load patterns and get split
    all_patterns = shuffle(
        list(get_all_pattern_fpaths()), random_state=args.random_seed
    )
    n_patterns = len(all_patterns)
    n_patterns_train = int(n_patterns * args.ratio_train_patterns)
    n_patterns_val = int(n_patterns * args.ratio_val_patterns)
    n_patterns_test = int(n_patterns * args.ratio_test_patterns)
    chosen_patterns_train = all_patterns[:n_patterns_train]
    chosen_patterns_val = all_patterns[
        n_patterns_train : n_patterns_train + n_patterns_val
    ]
    chosen_patterns_test = all_patterns[
        n_patterns_train
        + n_patterns_val : n_patterns_train
        + n_patterns_val
        + n_patterns_test
    ]

    # get tf datasets
    ds_train = get_ds_iterator_with_pattern_and_labels(
        img_fpaths_train,
        args.batch_size,
        args.input_image_height,
        args.input_image_width,
        chosen_patterns_train,
    )
    ds_val = get_ds_iterator_with_pattern_and_labels(
        img_fpaths_val,
        args.batch_size,
        args.input_image_height,
        args.input_image_width,
        chosen_patterns_val,
    )
    ds_train = ds_train.repeat(args.epochs)
    ds_val = ds_val.repeat(args.epochs)
    iterator_train = tf.compat.v1.data.make_one_shot_iterator(ds_train)
    iterator_val = tf.compat.v1.data.make_one_shot_iterator(ds_val)
    next_element_train = iterator_train.get_next()
    next_element_val = iterator_val.get_next()

    # prepare output folder
    out_folder_tensorboard = "/home/data/results/defense/%s/summary" % args.model_name
    out_folder_model = "/home/data/results/defense/%s/model" % args.model_name
    os.makedirs(out_folder_tensorboard, exist_ok=True)
    os.makedirs(out_folder_model, exist_ok=True)

    # network info
    C = params["defense"]["network_info"][args.model_name]
    C["meta_graph"] = f"/home/data/models/{args.model_name}/model.ckpt.meta"
    C["checkpoint"] = f"/home/data/models/{args.model_name}/"

    saver = tf.train.import_meta_graph(C["meta_graph"])

    g = tf.get_default_graph()

    with tf.Session() as sess:

        saver.restore(sess, tf.train.latest_checkpoint(C["checkpoint"]))
        feature_map = g.get_tensor_by_name(C["detection_features"])
        image_tensor = g.get_tensor_by_name("image_tensor:0")

        writer = tf.summary.FileWriter(out_folder_tensorboard, sess.graph)
        (
            train_summ,
            val_summ,
            test_summ,
            input_summ,
            loss_ph,
            accuracy_ph,
            loss_ph_val,
            accuracy_ph_val,
            loss_ph_test,
            accuracy_ph_test,
            input_summ_ph,
            mc_img_ph,
            mc_pfpath_ph,
        ) = make_training_writer()

        shutter_labels, logits, probs, accuracy, loss = add_shutter_head(
            feature_map,
            fully_conv=not args.use_dense,
            kernel_size=args.kernel_size,
            strides=args.strides,
            n_filters=args.num_filters,
            n_neurons_before_dense=C["n_neurons_before_dense"],
        )
        gd_step, opt, loss, trainables = add_shutter_training(loss)

        # do optimization
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        epoch = 0
        n_batches_train = len(range(0, len(img_fpaths_train), args.batch_size))

        while epoch < args.epochs:

            loss_train_, acc_train_ = 0, 0
            for j in tqdm(range(0, len(img_fpaths_train), args.batch_size)):

                images, labels, _ = sess.run(next_element_train)

                _, loss_train_, acc_train_ = sess.run(
                    [gd_step, loss, accuracy],
                    feed_dict={
                        image_tensor: images,
                        shutter_labels: pd.get_dummies(labels).values,
                    },
                )

                # Execute the summaries defined above
                if (j // args.batch_size % args.log_every) == 0:
                    summ_train_ = sess.run(
                        train_summ,
                        feed_dict={loss_ph: loss_train_, accuracy_ph: acc_train_},
                    )
                    writer.add_summary(
                        summ_train_, j // args.batch_size + (epoch * n_batches_train)
                    )

            epoch += 1

            if epoch % args.validate_every == 0:
                loss_val_, acc_val_, images_ = evaluate_validation(
                    sess, img_fpaths_val, next_element_val
                )
                summ_val_ = sess.run(
                    val_summ,
                    feed_dict={
                        loss_ph_val: loss_val_.mean(),
                        accuracy_ph_val: acc_val_.mean(),
                    },
                )
                writer.add_summary(summ_val_, epoch)
                input_summ_ = sess.run(input_summ, feed_dict={input_summ_ph: images_})
                writer.add_summary(input_summ_, epoch)

        print(f"Saving model to {out_folder_model}")
        saver.save(sess, os.path.join(out_folder_model, "m"))

        # do testing
        next_element_test = get_test_iterator(
            img_fpaths_test,
            args.batch_size,
            args.input_image_height,
            args.input_image_width,
            chosen_patterns_test,
        )
        loss_test_, acc_test_, mc_images_, mc_patterns_, result = evaluate_test(
            sess, img_fpaths_test, next_element_test
        )
        summ_test_ = sess.run(
            test_summ,
            feed_dict={
                loss_ph_test: loss_test_.mean(),
                accuracy_ph_test: acc_test_.mean(),
                mc_img_ph: mc_images_,
                mc_pfpath_ph: mc_patterns_,
            },
        )
        writer.add_summary(summ_test_, 0)
        writer.flush()

        print("All done, checking execution time now")

        normal_operation = [
            "detection_boxes:0",
            "detection_scores:0",
            "detection_classes:0",
            "num_detections:0",
        ]
        next_element_test = get_test_iterator(
            img_fpaths_test,
            1,
            args.input_image_height,
            args.input_image_width,
            chosen_patterns_test,
            repeat=2,
        )
        t1 = check_execution_time(
            sess, normal_operation, next_element_test, len(img_fpaths_test)
        )
        t2 = check_execution_time(
            sess,
            normal_operation + ["shutter_head/probs:0"],
            next_element_test,
            len(img_fpaths_test),
        )
        result["execution_time_pre"] = t1
        result["execution_time_post"] = t2

        with open(
            "/home/data/results/defense/%s/results.json" % args.model_name, "w"
        ) as outfile:
            json.dump(result, outfile)
