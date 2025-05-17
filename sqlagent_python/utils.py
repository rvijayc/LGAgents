import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path: str):

    img = mpimg.imread(image_path)

    plt.imshow(img)
    plt.axis("off")  # Hide axes for a clean display
    plt.show()

