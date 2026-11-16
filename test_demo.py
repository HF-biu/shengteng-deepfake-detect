import socket

def send_request(file_path='', input_type='image', host='127.0.0.1', port=54321):
    with open(file_path, 'rb') as file:
        file_data = file.read()
    # print(file_data)
    # print(len(file_data))

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    header = f'Content-Length: {len(file_data)}\nInput-Type: {input_type}'

    client.sendall(header.encode('utf-8') + b'\n\n\n\n' + file_data)

    response = client.recv(4096)
    response = response.decode('utf-8')
    response = float(response)
    client.close()
    return response

r1 = send_request('./demo_imgs/mms_media__072818852621__1.jpg', 'image')
print('response from server:', r1)
r2 = send_request('./1.mp4', 'video')
print('response from server:', r2)
r3 = send_request('./demo_imgs/mms_media__09470836001__1.jpg', 'image')
print('response from server:', r3)
