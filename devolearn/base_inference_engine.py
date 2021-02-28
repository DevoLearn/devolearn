class InferenceEngine():
    def __init__(self, model):
        self.model = model
        
    def download_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError
    
    def preprocess(self, x):
        raise NotImplementedError

    def deprocess(self, x):
        raise NotImplementedError

    def inference(self, x):
        raise NotImplementedError

    def __repr__(self):
        return self.model
