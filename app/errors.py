class RagError(Exception):
    """Erreur métier dont le message peut être renvoyé au client sans données privées."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail
