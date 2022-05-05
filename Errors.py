class FitError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __repr__(self):
        return f'FitError: {self.message}'