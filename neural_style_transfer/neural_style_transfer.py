import numpy as np
import keras
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras import models
import tensorflow.contrib.eager as tfe


class NeuralStyleTransfer():
    def __init__(self, number_of_epochs=1000, content_weight=1e3, style_weight=1e-2, verbose=False):
        self.number_of_epochs = number_of_epochs
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layers = ["block5_conv2"]
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
        self.verbose = verbose
        tf.enable_eager_execution()
        if(self.verbose == True):
            print "Eager Execution: {}".format(tf.executing_eagerly())

    def perform(self, content_file, style_file):
        return self._perform(content_file, style_file)

    def show_image(self, image):
        return self._show_image(image)

    def _load_image(self, filename):
        img = Image.open(filename)
        # Downsample Image using Image.ANTIALIAS
        img = img.resize((img.size[0], img.size[1]), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32)
        # Remove Alpha Channel if present.
        if(img.shape[2] == 4):
            img = img[:, :, :3]
        # Convert Image Shape to (1, X, X, X)
        img = np.expand_dims(img, axis=0)
        return img

    def _preprocess_image(self, filename):
        img = self._load_image(filename)
        # Preprocess Image to VGG19 Input format (Subtracts Input by VGG Mean and
        # uses cv2 to open the image which uses BGR format)
        #im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
        #im[:,:,0] -= 103.939
        #im[:,:,1] -= 116.779
        #im[:,:,2] -= 123.68
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def _deprocess_image(self, processed_image):
        x = processed_image.copy()
        if(len(x.shape) == 4):
            x = np.squeeze(x)
        assert(len(x.shape) == 3)

        # Deprocess Image by adding back the VGG Mean.
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.680

        # Convert from BGR to RGB.
        x = x[:, :, ::-1]
        # Make sure minimum value of a pixel is 0 and maximum value of a pixel is 255.
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def _get_model(self):
        vgg_model = tf.keras.applications.vgg19.VGG19(include_top = False, weights='imagenet')
        vgg_model.trainable = False
        if(self.verbose == True):
            print vgg_model.summary()

        # Get Outputs of Content Layers and Style Layers
        content_outputs = [vgg_model.get_layer(layer).output for layer in self.content_layers]
        style_outputs = [vgg_model.get_layer(layer).output for layer in self.style_layers]

        model_outputs = style_outputs + content_outputs
        return models.Model(vgg_model.inputs, model_outputs)

    def _get_activations(self, model, content_file, style_file):
        content_image = self._preprocess_image(content_file)
        style_image = self._preprocess_image(style_file)

        content_outputs = model(content_image)
        style_outputs = model(style_image)

        # Get activations of respective layers. content_layer[0] is done ton convert
        # list shape of (1, X, X, X) to (X, X, X)
        content_image_activations = [content_layer[0] for content_layer in content_outputs[len(self.style_layers):]]
        style_image_activations = [style_layer[0] for style_layer in style_outputs[:len(self.style_layers)]]

        return content_image_activations, style_image_activations

    def _content_loss_computation(self, content_image_activations, generated_image_activations):
        # Content Loss Computation
        return tf.reduce_mean(tf.square(content_image_activations - generated_image_activations))

    def _gram_matrix_computation(self, input_activations):
        # Gram Matrix Computation
        channels = input_activations.shape[-1]
        activations = tf.reshape(input_activations, [-1, channels])
        number_of_activations = activations.shape[0]
        gram_matrix = tf.matmul(activations, activations, transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(number_of_activations, tf.float32)
        return gram_matrix

    def _style_loss_computation(self, gram_matrix_style_image, gram_matrix_generated_image):
        # Style Loss Computation
        return tf.reduce_mean(tf.square(gram_matrix_style_image - gram_matrix_generated_image))

    def _compute_overall_loss(self, model, loss_weights, generated_image, content_image_activations, style_image_activations):
        # While training, first the style loss decreases rapidly to some small value.
        #After the content loss starts decreasing, the style loss either decreases
        #much more slowly or fluctuates.
        #Weights should be initialized such that the content loss is significantly
        #greater than the style loss at this point. Otherwise network will not learn
        #any of the content.
        content_weight, style_weight = loss_weights

        model_outputs = model(generated_image)

        # Get respective activations of generated image
        generated_image_activations_content = [content_layer[0] for content_layer in model_outputs[len(self.style_layers):]]
        generated_image_activations_style = [style_layer[0] for style_layer in model_outputs[:len(self.style_layers)]]
        # Convert style image and generated image into their respective gram matrices
        gram_matrices_style_image = [self._gram_matrix_computation(activation) for activation in style_image_activations]
        gram_matrices_generated_image = [self._gram_matrix_computation(activation) for activation in generated_image_activations_style]

        style_loss = 0
        content_loss = 0
        # Compute the content loss
        weight_per_content_layer = 1.0 / float(len(self.content_layers))
        for a, b in zip(content_image_activations, generated_image_activations_content):
            content_loss = content_loss + (weight_per_content_layer * self._content_loss_computation(a, b))

        # Compute the style loss
        weight_per_style_layer = 1.0 / float(len(self.style_layers))
        for a, b in zip(gram_matrices_style_image, gram_matrices_generated_image):
            style_loss = style_loss + (weight_per_style_layer * self._style_loss_computation(a, b))

        content_loss = content_loss * content_weight
        style_loss = style_loss * style_weight
        overall_loss = content_loss + style_loss

        return overall_loss, content_loss, style_loss

    def _compute_gradients(self, parameters):
        with tf.GradientTape() as g:
            losses = self._compute_overall_loss(**parameters)
        overall_loss = losses[0]
        # Compute the Gradient dJ(G) / d(G) using tf.GradientTape()
        return g.gradient(overall_loss, parameters['generated_image']), losses

    def _perform(self, content_file, style_file):
        # Run Neural Style Transfer.
        model = self._get_model()
        for layer in model.layers:
            layer.trainable = False
        content_image_activations, style_image_activations = self._get_activations(model, content_file, style_file)
        generated_image = self._preprocess_image(content_file)
        generated_image = tfe.Variable(generated_image, dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
        loss_weights = (self.content_weight, self.style_weight)
        final_loss, final_image = np.inf, None
        parameters = {
        "model" : model,
        "loss_weights": loss_weights,
        "generated_image": generated_image,
        "content_image_activations": content_image_activations,
        "style_image_activations": style_image_activations
        }
        num_rows = 2
        num_columns = 5
        display_interval = self.number_of_epochs / (num_rows*num_columns)

        norm_means = np.array([103.939, 116.779, 123.680])
        min_value = -norm_means
        max_value = 255 - norm_means
        intermediate_images = []
        for i in range(self.number_of_epochs):
            gradients, losses = self._compute_gradients(parameters)
            overall_loss, content_loss, style_loss = losses
            optimizer.apply_gradients([(gradients, generated_image)])
            clipped_image = tf.clip_by_value(generated_image, min_value, max_value)
            generated_image.assign(clipped_image)

            if overall_loss < final_loss:
                final_loss = overall_loss
                final_image = generated_image.numpy()
                final_image = self._deprocess_image(final_image)
            if(self.verbose == True):
                print "Epoch Number %d : Completed" % (i+1)
            if i % display_interval == 0:
                intermediate_image = generated_image.numpy()
                intermediate_image = self._deprocess_image(intermediate_image)
                intermediate_images.append(intermediate_image)

        # Show Intermediate Images
        plt.figure(figsize=(14,4))
        for i,intermediate_image in enumerate(intermediate_images):
            plt.subplot(num_rows,num_columns,i+1)
            plt.imshow(intermediate_image)
            plt.xticks([])
            plt.yticks([])
        plt.savefig("Output_Images/Intermediate_Images.jpg")
        plt.show()
        plt.clf()
        plt.close()
        return final_image

    def _show_image(self, image):
        # Display the final image.
        plt.axis('off')
        plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
        plt.imshow(image)
        plt.savefig("Output_Images/Style_Transfer.jpg", bbox_inches=None, pad_inches=0)
        plt.show()
        return
