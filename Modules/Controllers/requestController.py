class reqController:

    def __init__(self,
    currentStatus = {"Btn1_status" : False , "Btn2_status" : False , "Btn3_status" : False , "esp_connected" : True},
    switch_autoControl = {"Btn1_autoControl" : True , "Btn2_autoControl" : True , "Btn3_autoControl" : True}):
        self.currentStatus = currentStatus
        self.switch_autoControl = switch_autoControl
    
    def getCurrentStatus(self):
        return (self.currentStatus)
    
    def getControlStatus(self):
        return (self.switch_autoControl)

    def setCurrentStatus(self,data):
        for k,v in data.items():
            # if ((k != 'showProcessedVideo')):
            #     if((data[k] != self.currentStatus[k])):
            #         self.currentStatus[k] = data[k]
            self.currentStatus[k] = data[k]
        print(self.currentStatus)
        return (self.currentStatus)
        
    
    def setControlStatus(self,data):
        for k,v in data.items():
            if(data[k] != self.switch_autoControl[k]):
                self.switch_autoControl[k] = data[k]
        print(self.switch_autoControl)
        return (self.switch_autoControl)
    
    def controlAutoStatus(self,object_count):
        if (object_count > 20):
            btn1_status = True if (self.switch_autoControl['Btn1_autoControl']) else self.currentStatus['Btn1_status']
            btn2_status = True if (self.switch_autoControl['Btn2_autoControl']) else self.currentStatus['Btn2_status']
            Btn3_status = True if (self.switch_autoControl['Btn3_autoControl']) else self.currentStatus['Btn3_status'] 
        else :
            btn1_status = False if (self.switch_autoControl['Btn1_autoControl']) else self.currentStatus['Btn1_status']
            btn2_status = False if (self.switch_autoControl['Btn2_autoControl']) else self.currentStatus['Btn2_status']
            Btn3_status = False if (self.switch_autoControl['Btn3_autoControl']) else self.currentStatus['Btn3_status']
        
        self.currentStatus = {"Btn1_status" : btn1_status , "Btn2_status" : btn2_status , "Btn3_status" : Btn3_status}