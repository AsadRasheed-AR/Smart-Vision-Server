import threading
import requests
import time


class espController:
    
    def __init__(self,rc_obj):
        self.rc_obj = rc_obj
        self.esp_Base_Url = "https://192.168.4.1:80"
        # self.esp_setStatus_url = "https://192.168.4.1:80/setCurrentStatus"
        self.esp_setStatus_url = "http://192.168.4.1:80/setCurrentStatus"
    
    def set_esp_status(self):
        while(True):
            time.sleep(2.0)
            try:
                res = requests.post(self.esp_setStatus_url,json=self.rc_obj.currentStatus)
                if (res.status_code != 200):
                    self.rc_obj.currentStatus["esp_connected"] = False
                elif (not self.rc_obj.currentStatus["esp_connected"]):
                    self.rc_obj.currentStatus["esp_connected"] = True
            except requests.exceptions.ConnectionError:
                print('Connection Refused')
                self.rc_obj.currentStatus["esp_connected"] = False

            except Exception as e:
                print(e)
            finally :
                print(self.rc_obj.currentStatus)
            # except expression as identifier:
            #     pass
    
    def startAsyncOperations(self):
        async_set_esp_status=threading.Thread(target=self.set_esp_status)
        async_set_esp_status.daemon = True
        async_set_esp_status.start()

        