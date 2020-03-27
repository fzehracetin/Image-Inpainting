import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
import time

from inpaint_model import InpaintCAModel


if __name__ == "__main__":
    start_time = time.time()

    FLAGS = ng.Config('inpaint.yml')
    image_height = 256
    image_width = 256
    checkpoint_dir = "C:/Users/fzehr/PycharmProjects/image_inpainting/model_logs/release_places2_256"
    model = InpaintCAModel()
    test_dir = "C:/Users/fzehr/PycharmProjects/image_inpainting/test/"
    mask = cv2.imread("C:/Users/fzehr/PycharmProjects/image_inpainting/mask.jpg")

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, image_height, image_width * 2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []

    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')
    f_count = len(os.listdir(test_dir))
    i = 1
    for filename in os.listdir(test_dir):
        print("{}/{}".format(i, f_count))
        image = cv2.imread(test_dir + filename)
        mask_tmp = mask

        assert image.shape == mask_tmp.shape
        h, w, _ = image.shape
        # print('Shape of image: {}'.format(image.shape))
        grid = 8
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask_tmp = mask_tmp[:h // grid * grid, :w // grid * grid, :]
        image = np.expand_dims(image, 0)
        mask_tmp = np.expand_dims(mask_tmp, 0)

        input_image = np.concatenate([image, mask_tmp], axis=2)
        filename = filename.split('.')
        output_name = test_dir + filename[0] + "_output_1." + filename[1]

        result = sess.run(output, feed_dict={input_image_ph: input_image})
        cv2.imwrite(output_name, result[0][:, :, ::-1])
        i += 1

    print("Total time: ", time.time() - start_time)
