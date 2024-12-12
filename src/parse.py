import os
import numpy as np
from PIL import Image

def remove_green(img, threshold=50):
    target_green = [0, 255, 0]

    green_mask = (img[:, :, 1] > img[:, :, 0] + threshold) & \
                 (img[:, :, 1] > img[:, :, 2] + threshold) & \
                (img[:, :, 0] < 200) & (img[:, :, 2] < 200)

    modified_array = img.copy()
    modified_array[green_mask] = target_green

    return modified_array

def read_img(path, crop_top=False, crop_bottom=False, apply_remove_green=False, resize=(96,96), grayscale=False):
    if resize[0] != resize[1]:
        print("[E] Resize dimensions must be equal")
        exit()
    
    if grayscale:
        img = Image.open(path).convert('L')
    else:
        img = Image.open(path).convert('RGB')
    img = img.resize(resize)

    crop = resize[0] // 8
    
    if crop_top and crop_bottom:
        print("[E] Cannot crop top and bottom at the same time")

    if crop_top:
        img = img.crop((0, resize[1]-crop, resize[0], resize[1]))
    elif crop_bottom:
        # img = img.crop((0, resize[1] - crop, resize[0], resize[1]))
        img = img.crop((0,0,resize[0],resize[1]-crop))
    
    img = np.array(img)
    if apply_remove_green:
        if grayscale:
            print("[E] Cannot remove green from grayscale image")
            exit()
        img = remove_green(img)
    return img

def show_img(img):
    Image.fromarray(img).show()

def save_img(img, path):
    Image.fromarray(img).save(path)

def load_data(path):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.png'):
                X_train.append(read_img(os.path.join(root, file), resize=(32,32), grayscale=True, crop_top=True))
                y_train.append(root.split('/')[-1])
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.png'):
                X_test.append(read_img(os.path.join(root, file)))
                y_test.append(root.split('/')[-1])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data('../Dataset')

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # print("Sample train images:", X_train[-1])
    # print("Sample test images:", X_test[-1])
    # print("Sample train labels:", y_train[-1])
    # print("Sample test labels:", y_test[-1])

    save_img(X_train[-1], "./sample.png")
    # save_img()
