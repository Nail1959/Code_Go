<<<<<<< HEAD
__all__ = [
    'Agent',
]


# tag::agent[]
class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()
# end::agent[]

    def diagnostics(self):
        return {}
=======
__all__ = [
    'Agent',
]


# tag::agent[]
class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()
# end::agent[]

    def diagnostics(self):
        return {}
>>>>>>> 9a1c796396bfb5163e70f29fda90217dd89512e3
