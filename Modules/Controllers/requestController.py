class reqController:

    def __init__(self,
    currentStatus = {"Btn1_status" : True , "Btn2_status" : True , "Btn3_status" : True},
    switch_autoControl = {"Btn1_autoControl" : True , "Btn2_autoControl" : False , "Btn3_autoControl" : True}):
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
