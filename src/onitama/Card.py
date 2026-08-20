class Card():
    def __init__(self, name, idx, moveset):
        self.name = name
        self.idx = idx
        self.moveset = moveset
        self.matrix = self.create_matrix()

    def create_matrix(self):
        # Initialise matrix for containing possible moves
        matrix = [[" "," "," "," "," "] for i in range(4)]
        matrix[1][2] = "@"
        
        # Loop through moves, replace corresponsing matrix entry with "#"
        for move in self.moveset:
            matrix[move[1]+1][move[0]+2] = "#"

        # Initialise full_matrix with first line (spaced card name)
        full_matrix = [
            (9-len(self.name))//2*" "+f"{self.name}"+-(-(9-len(self.name))//2)*" "
        ]

        # Prepare string to make f string list join work
        space = " "

        # successively fill out the rest of full_matrix
        for i in range(4):
            full_matrix.append(f"{space.join(matrix[i])}")
        
        return full_matrix