from transformers import AutoProcessor, BlipForConditionalGeneration

class model:
    def __init__(self,pretrain):
        self.model = BlipForConditionalGeneration.from_pretrained(pretrain)
        self.processor = AutoProcessor.from_pretrained(pretrain)
        
    def return_model(self):
        return {"model" : self.model, "processor" : self.processor}




