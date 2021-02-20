
### working solution 
class Solution:
    
    all_digits = ["{}".format(i) for i in range(1,10)]
    
    def generateCandidates(self, current:List[str]) -> List[str]:
        # drop the 
        all_digits_set = set(self.all_digits) 
        current_filtered_set = set(list(filter(lambda a: a != '.', current)))
        #print(f'current are {current_filtered_set}')
        candidates = list(all_digits_set.difference(current_filtered_set))
        return candidates
    
    def generateNextAvailablePositions(self, row:int, col:int, board: List[List[str]]) -> (int,int):
        #print(f'searching starting point row {row} and column {col}')
        
        while row < len(board):
            if board[row][col] == '.':
                #print(f'next available position is row {row} and column {col}')
                return (row,col)
            else:
                col = col + 1
                if col > 8:
                    row = row + 1
                    col = 0
        
        return (-1,-1)
        
    def is_candidate_valid_column(self,c:str, r:int,col:int,board: List[List[str]]) -> bool:
        
        for rr in range(0,len(board)):
            if board[rr][col] == c :
                #print(f'candidate {c} not valid, has occurred in previous rows {board[rr]} ')
                return False
        return True
    
    def is_candidate_valid_row (self,c:str, r:int, board: List[List[str]]) -> bool:
        
        for cc in range(0,len(board[r])):
            if board[r][cc] == c:
                #print(f'candidate {c} not valid, has occurred in the same row {board[r]}')
                return False
        return True
                
    def is_candidate_valid_grid(self,c:str, r:int, col:int,board: List[List[str]]) -> bool:
        
        r_start = floor(r / 3) * 3 # upper row boundary for grid 
        v_start = floor(col / 3) * 3 # left column boundary grid of 
        
        subboard = [l[v_start:(v_start+3)] for l in board[r_start:(r_start+3)]]
        value_found = any(c in sublist for sublist in subboard)
        #if  value_found:
            #print(f'candidate {c} not valid, has occurred in the grid {subboard} ')
        return (not value_found)
            
    def is_candidate_valid(self,c:str, r:int,col:int,board: List[List[str]]) -> bool:
        # no need to check similar row
        # need to check previous row same column
        # also the 3x3 grid 
        
        return self.is_candidate_valid_column(c,r,col,board) and self.is_candidate_valid_grid(c,r,col,board) and self.is_candidate_valid_row(c,r,board)
    
    
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.all_digits = ["{}".format(i) for i in range(1,len(board)+1)]
        #print(f'all candidates are {self.all_digits}')
        
        self.sudoku(board,0,0) # starting with the first rows 
    
    def sudoku(self, board: List[List[str]], r: int, col: int) -> bool:
        
        # generate next positions to fill 
        rr,cc = self.generateNextAvailablePositions(r,col, board) # get the next positions to fill
        
        n = len(board)
        
        if rr is -1:
            #print(f'current board is {board}')
            #print(f'searching starting point row {r} and column {col}')
            #print(f'no available position to fill, solution reached !! ')
            return True
        
        else: 
            # generate all candidates that can fill the open slot
            candidates = self.generateCandidates(board[rr])
            
            for c in candidates:
                #print(f'current position is row {rr}, column {cc}, candidate is {c}')
                if self.is_candidate_valid(c,rr,cc, board):
                    board[rr][cc] = c
                    #print(f'successfully put {c} in row {rr} and column {cc}')
                    if self.sudoku(board,rr,cc) is False: # can not reach a solution
                       board[rr][cc] = '.'  # back track, remove current candidate move to the next candidate
                       #print(f'removing value from row {rr} and column {cc}')
                    else:
                        return True 
            return False # means can not fill  