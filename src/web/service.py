class TitleClassify:
    def __init__(self,predictor):
        self.predictor = predictor

    def predict( self,text:str | list ):
        return self.predictor.predict(text)