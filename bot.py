import requests
import misc
from time import sleep

import shutil
import os

from predict import save_image, colorize_mask
from model import DeepLab
import torch
from catalyst.utils import imread
import torch.nn.functional as F
import albumentations as albu
from albumentations.pytorch import ToTensor


token = misc.token
password = misc.war_token
admin = misc.admin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLab(num_classes=18, pretrained=False)
model.load_state_dict(torch.load(misc.model_path, map_location=device)['model_state_dict'])
model.to(device)
model.eval()

URL = f"https://api.telegram.org/bot{token}/"
FILE_URL = f"https://api.telegram.org/file/bot{token}/"
GET_FILE_URL = f"https://api.telegram.org/bot{token}/getFile?file_id="

try:
    with open ('user_id.txt', 'r+') as file:
        user_ids = set()
        for user in file:
            user_ids.add(user.rstrip('\n'))
except FileNotFoundError:
    user_ids = set()

def evaluate_image(image_path):
    image = imread(image_path)
    valid_transformation = albu.Compose([albu.Normalize(), ToTensor()])
    im = valid_transformation(image=image)["image"].unsqueeze(0)
    prediction = model(im.to(device))
    prediction = prediction.squeeze(0).detach().cpu().numpy()

    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
    mask_path = save_image(image, prediction, image_path.replace(".", "_new."))
    return image_path.replace(".", "_new."), mask_path


def get_file(image):
    if isinstance(image, list):
        file_id = image[-1]['file_id']
    else:
        file_id = image.get("file_id", 0)
        file_type = image['mime_type']
        if file_type not in('image/jpeg', 'image/png'):
            return -1

    server_path = get_dict(GET_FILE_URL, file_id)["result"]['file_path']
    load_url = FILE_URL + server_path
    local_path = 'temp/' + server_path.rsplit("/")[-1]

    response = requests.get(load_url , stream=True)
    with open(local_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    print(f'save {local_path}')
    return local_path

def write_users(id, path='user_id.txt'):
    with open(path, 'a') as f:
        f.writelines(str(id) + '\n')

def send_message(chat_id, text='Wait a second, please...'):
    url = URL + f'sendmessage?chat_id={chat_id}&text={text}'
    requests.get(url)

def send_image(chat_id, image_path='', text=None):
    if text:
        send_message(chat_id, text)
    data = {'chat_id': chat_id}
    files = {'photo': open(image_path, 'rb')}
    requests.post(f"{URL}sendPhoto", data=data, files=files)


def get_dict(url, events):
    url = url + events
    r = requests.get(url)
    return r.json()

def get_message(last_id=None):
    global upd_id, upd_counter
    data = get_dict(URL, 'getupdates')

    try:
        last_result = data['result'][-1]
    except IndexError:
        return None

    upd_id = last_result["update_id"]
    if last_id == upd_id:
        return None

    upd_counter += 1

    chat_id = str(last_result['message']['chat']['id'])


    if chat_id not in user_ids:
        message_text = last_result['message'].get('text')
        if message_text == password:
            user_ids.add(chat_id)
            write_users(chat_id)
        else:
            send_message(chat_id, text='Enter password')
            return None


    image = last_result['message'].get('document', 0) or last_result['message'].get('photo', 0)

    if not image:
        send_message(chat_id, text='Please send image')
        return None
    send_message(chat_id)

    path = get_file(image)


    if path == -1:
        send_message(chat_id, "Unsupported file type, please choose another type (jpg, png)")
        return

    image_path, mask_path = evaluate_image(path)
    send_image(chat_id, image_path)
    send_image(chat_id, mask_path)

    username = last_result["message"]["chat"]["username"]

    print(username)
    if username != admin:
        send_message(392426836, f'{username} send image')
    return None



def refresh_extrad():
    upd_id = get_dict(URL, 'getupdates')['result'][-1]["update_id"]
    url = URL + f"getupdates?offset={upd_id + 1}"
    requests.get(url)
    return  upd_id



if __name__ == "__main__":
    shutil.rmtree('temp', ignore_errors=True)
    os.makedirs('temp')

    try:
        upd_id = refresh_extrad()
        print('REFRESH')
    except IndexError:
        upd_id = 1

    upd_counter = 0

    while True:
        get_message(upd_id)
        if upd_counter == 90:
            upd_id = refresh_extrad()
            upd_counter = 0
            print('reset')
        sleep(0.2)
