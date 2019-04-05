from neural_style_transfer import NeuralStyleTransfer

content_file = "Input_Images/Amsterdam.jpg"
style_file = "Input_Images/Starry_Night.jpg"

nst = NeuralStyleTransfer(verbose=True)
image = nst.perform(content_file, style_file)
nst.show_image(image)
