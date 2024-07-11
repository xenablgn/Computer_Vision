# Constants
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASS_NAMES_LABEL = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}
NB_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (224, 224)
TRAIN_PATH = '../Data/images/train'
TEST_PATH = '../Data/images/test'
IMAGE_CHANNEL = 3  # RGB images
BATCH_SIZE = 32
EPOCHS = 25
EPOCH_TR=50
MODEL_SAVE_PATH = '../Models/cv_final_basic_cnn.h5'
MODEL_SAVE_PATH_TR='../Models/cv_final_advanced.h5'
DATA_DIR = '../Data/images'  # Adjust this path according to your data directory structure
PIC_SIZE = 48
FOLDER_PATH = "../Data/images/"