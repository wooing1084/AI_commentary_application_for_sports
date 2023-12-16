class Player:
    def __init__(self, ID, team, color):
        self.ID = ID
        self.team = team
        self.color = color
        self.bboxs = {}
        self.previous_bb = None
        self.positions = {}
        self.has_ball = False