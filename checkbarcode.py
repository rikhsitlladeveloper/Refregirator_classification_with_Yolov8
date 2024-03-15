#!/usr/bin/env python3
import socket
import json


class BarCodeCheck:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.samsung_rb = []
        self.samsung_rt = []
        self.bar_codes_path = "bar_codes.json"
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def get_barcode(self) -> None:
        with open(self.bar_codes_path, 'r') as file:
            data = json.load(file)
        self.samsung_rb = data["Samsung RB"]
        self.samsung_rt = data["Samsung RT"]

    def check_barcode(self, barcode:str) -> str:
        if len(barcode) >= 4:
            model = barcode[:4]
            if model in self.samsung_rt:
                return "Samsung RT"
            elif model in self.samsung_rb:
                return "Samsung RB"
            else:
                return "Not Specified Barcode"
        else:
            return "Not Specified Barcode"

    def start_tcp_server(self) -> None:
        try:
            self.server_socket.bind((self.host, self.port))

            self.server_socket.listen(5)

            print(f"TCP server is listening on {self.host}:{self.port}")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.server_socket.close()

    def get_tcp_server_data(self) -> str:
        client_socket, client_address = self.server_socket.accept()

        print(f"Accepted connection from {client_address}")

        try:
            data = client_socket.recv(1024)

            if data:
                decoded_data = data.decode('utf-8').replace('\r\n', '')
                print(f"Received data: {decoded_data}")
                return decoded_data
            else:
                return ""

        except KeyboardInterrupt:
            print('Socket close')
            client_socket.close()
            self.server_socket.close()

        except Exception as e:
            print(f"Error while handling client connection: {e}")

        finally:
            client_socket.close()
            self.server_socket.close()


if __name__ == "__main__":
    host = "10.10.21.216"
    port = 8001
    barcode = BarCodeCheck(host,port)
    barcode.get_barcode()
    barcode.start_tcp_server()
    while True:
        code = barcode.get_tcp_server_data()
        model = barcode.check_barcode(code)