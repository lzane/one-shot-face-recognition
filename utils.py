from skimage import io

def load_image(path):
    return io.imread(path,img_num=0).astype('uint8')