import requests
import json
import time

def send_message(to, text, chat_id=1):
    data = {
        'botToken': to,
        'from': 'TestClient',
        'chat': {
            'type': 'text',
            'id': chat_id,
        },
        'date': int(time.time()),
        'text': text
    }
    requests.post('http://localhost:9000/sendMessage', json=data)

def fetch_messages(token):
    data = {
        'token': token
    }
    
    req = requests.post('http://localhost:9000/getUpdates', json=data)
    return req.json()

send_message('XXXXXXXXXXXXXXXXX', '/start')
time.sleep(1)

print(fetch_messages('XXXXXXXXXXXXXXXXX'))
