import sys
import parse
import numpy as np
from PIL import Image
from tensorflow import keras

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)

def play(env: gym.Env, model, lenet=False, crop_bottom=False):
    seed = 2000

    obs, _ = env.reset(seed=seed)

    action0 = 0
    for i in range(50):
        obs, _, _, _, d = env.step(action0)


    done = False
    SIZE = 32 if lenet else 96
    crop = SIZE // 8
    while not done:

        if lenet:
            img = Image.fromarray(obs).convert('L')
            img = img.resize((SIZE,SIZE))
            obs = np.array(img)
            obs = obs / 255.0
            obs = np.expand_dims(obs, axis=2)

        else:
            obs = parse.remove_green(obs)
            obs = obs / 255.0

        # Image.fromarray(obs).save("tmp.png")
        # env.close()
        # exit()
        if crop_bottom:
            obs = obs[:-crop,:,:]
        obs_input = np.expand_dims(obs, axis=0)

        p = model.predict(obs_input)
        action = np.argmax(p)

        obs, _, terminated, truncated, d = env.step(action)
        done = terminated or truncated


env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human'
}

env_name = 'CarRacing-v3'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

LENET = False
CROP_BOTTOM = False
# Load your trained model
# model_path = '../Hyp_Tuning/LeNet_e10_b32_lr5/cnn_model.keras'  # fortissima, con 360 prima del curvone
# model_path = '../Hyp_Tuning/LeNet_e30_b64/cnn_model.keras'  # scema ma fa bene il curvone
# model_path = f'../Hyp_Tuning/CNN_e10_b32/cnn_model.keras'   # non frena, va al contrario poi

# NOT CROPPED
# model_path = f'../Report/CNN_e10_b128_lr5/cnn_model.keras'  # non frena, va al contrario poi
# model_path = f'../Report/CNN_e10_b256_lr5/cnn_model.keras'  # fortissima, con 360 sul curvone
# model_path = f'../Hyp_Tuning/CNN_e10_b32_lr5_RMSprop/cnn_model.keras'   # si perde al curvone
# model_path = f'../Hyp_Tuning/CNN_e10_b128_lr5_RMSprop/cnn_model.keras'    # non frena, va al contrario poi

# model_path = f'../Hyp_Tuning/Another_CNN/' + other[5] + '/cnn_model.keras'    # completato la corsa!!!!!
# model_path = f'../Hyp_Tuning/Another_CNN/' + other[11] + '/cnn_model.keras'     # si ribalta

# model_path = f'../Report/ImageNet_e10_b32_lr1/cnn_model.keras'
other = ['CNN_e10_b32_lr5', 'CNN_e10_b64_lr1', 'CNN_e10_b64_lr5', 'CNN_e30_b64_lr7', 'CNN_e30_b32_lr5', 'CNN_e30_b32_lr1', 'CNN_e30_b64_lr5', 'CNN_e10_b64_lr7', 'CNN_e10_b32_lr1', 'CNN_e10_b32_lr7', 'CNN_e30_b32_lr7', 'CNN_e30_b64_lr1']
model_path = f'../Hyp_Tuning/Another_CNN/' + other[5] + '/cnn_model.keras'
model = keras.models.load_model(model_path)
print("Model loaded from:", model_path)

play(env, model, lenet=LENET, crop_bottom=CROP_BOTTOM)

env.close()

