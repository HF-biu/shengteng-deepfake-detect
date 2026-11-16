
import os
import requests
import json
from PIL import Image
from io import BytesIO

# def fetch_image_data():
url = os.environ['MEDIA_API_URL']
headers = {
    "Content-Type": "application/json"
}
params = {
    "currentPage": 1,
    "pageSize": 9999,
    "msgType": "2",
    "fileAttribute": ["2"],
    "startTime": "2025-02-26 00:00:00",
    "endTime": "2025-02-27 00:00:00"
}

response = requests.post(url, headers=headers, json=params)

if response.status_code == 200:
    data = response.json()

    current_page_size = len(data["data"]["dataList"])

    data_list = data.get('data', {}).get('dataList', [])
    for element in data_list:
        objectId = element['objectId']
        webCallbackUrl = element['webCallbackUrl']
        element_response = requests.get(webCallbackUrl, stream=True)
        if element_response.status_code == 200:
            img = BytesIO(element_response.content)
            # print(img)
