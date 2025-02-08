import matplotlib.image as mpimg
from matplotlib.pyplot import imshow, axis, show
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

image_input = input("Enter the path of the image:")

images = model(image_input)

annotated_image = images[0].plot() 

imshow(annotated_image)

axis('off')

show()
