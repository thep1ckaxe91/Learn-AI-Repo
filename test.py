from PIL import Image
img = Image.open('image.png')
gray = img.convert('L')
gray.show()