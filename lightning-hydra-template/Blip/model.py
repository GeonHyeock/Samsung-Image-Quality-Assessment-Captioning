from transformers import AutoProcessor, BlipForConditionalGeneration

class model:
    def __init__(self,pretrain):
        """
            text_decoder:
                bert:
                    embeddings
                        embeddings
                        encoder
                cls:
                    prediction:
                        transform
                        decoder
        """
        
        self.model = BlipForConditionalGeneration.from_pretrained(pretrain)
        self.processor = AutoProcessor.from_pretrained(pretrain)

        for name, childs in self.model.named_children():
            if name == "vision_model":
                for param in childs.parameters():
                    param.requires_grad = False

            elif name == "text_decoder":
                for param in childs.bert.parameters():
                    param.requires_grad = False
        
    def return_model(self):
        return {"model" : self.model, "processor" : self.processor}




