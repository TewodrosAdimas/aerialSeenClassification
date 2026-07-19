import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
        pred_index=None):


    grad_model = tf.keras.models.Model(
        [
            model.inputs
        ],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)


        if pred_index is None:
            pred_index = tf.argmax(predictions[0])


        class_channel = predictions[:, pred_index]


    grads = tape.gradient(
        class_channel,
        conv_outputs
    )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = conv_outputs @ pooled_grads[...,tf.newaxis]

    heatmap = tf.squeeze(heatmap)


    heatmap = np.maximum(
        heatmap,
        0
    )


    heatmap /= np.max(heatmap)


    return heatmap.numpy()



def overlay_gradcam(
        img,
        heatmap,
        alpha=0.4):


    heatmap=cv2.resize(
        heatmap,
        (img.shape[1],img.shape[0])
    )


    heatmap=np.uint8(
        255*heatmap
    )


    heatmap=cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    output = heatmap*alpha + img


    return np.uint8(output)