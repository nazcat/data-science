import numpy as np
import matplotlib.pyplot as plt


def gol_step( U ):
    
    nx   = U.shape[0] # return a list of number of rows/columns, indexed at 0
    
    idx1 = np.argwhere( U == 1 ) # alive cells
    idx0 = np.argwhere( U == 0 ) # dead cells
    
    U_new = U.copy() # keep original U, copy for iterations
    
    U_ghosted = np.zeros( [nx+2 , nx+2] ) # makes grid size +2 larger vertically and horizontally (32x32 becomes 34x34)
    
    U_ghosted[1:-1 , 1:-1] = U.copy() # copies values in the 32x32 grid
    # U_ghosted[0,:] = 0
    # U_ghosted[-1,:] = 0
    # U_ghosted[:,0] = 0
    # U_ghosted[:,-1] = 0
    
    # U_ghosted[0,1:-1]  = U[-1,:].copy() # copies bottom row in the 32x32 grid into the top of the 34x34 grid
    # U_ghosted[-1,1:-1] = U[0,:].copy() # copies top row in the 32x32 grid into the bottom of the 34x34 grid
    # U_ghosted[1:-1,0]  = U[:,-1].copy() # copies right column in the 32x32 grid into the left of the 34x34 grid
    # U_ghosted[1:-1,-1] = U[:,0].copy() # copies left column in the 32x32 grid into the right of the 34x34 grid
    # U_ghosted[0,0]     = U[-1,-1].copy() # copies bottom right corner in the 32x32 grid into the upper left corner of the 34x34 grid
    # U_ghosted[-1,-1]   = U[0,0].copy() # copies upper left corner in the 32x32 grid into the bottom right corner of the 34x34 grid
    # U_ghosted[-1,0]    = U[0,-1].copy() # copies upper right corner in the 32x32 grid into the bottom left corner of the 34x34 grid
    # U_ghosted[0,-1]    = U[-1,0].copy() # copies bottom left corner in the 32x32 grid into the upper right corner of the 34x34 grid
    
    for i in range( idx1.shape[0] ): #function for live cells
        
        I = idx1[i][0] + 1  # referencing live cells from the ghosted cells 34x34
        J = idx1[i][1] + 1
        
        square =  U_ghosted[ (I-1) : (I+2) ,\
                             (J-1) : (J+2) ] # references the 3x3 cell around a single live cell
        
        num1   = square.sum() - 1 # subtract 1 since center is 1, identifies number of live neighboring cells
        
        if ( num1 < 2 ):
            U_new[ idx1[i][0],idx1[i][1] ] = 0 # if number of live neighbors is <2 then cell dies
            
        elif ( num1 <= 3 ):
            pass # keep cell alive if there are 2 or 3 neighboring live cells
        
        else:
            U_new[ idx1[i][0],idx1[i][1] ] = 0 # kills cells
            
    for i in range( idx0.shape[0] ): # function for dead cells
        
        I = idx0[i][0] + 1 # referencing dead cells from the ghosted cells 34x34
        J = idx0[i][1] + 1
        
        square =  U_ghosted[ (I-1) : (I+2) ,\
                             (J-1) : (J+2)  ] # references the 3x3 cell around a single dead cell
        num1   = square.sum() # Don't subtract anything since center is zero
        
        if ( num1 == 3 ):
            U_new[ idx0[i][0],idx0[i][1] ] = 1 # revive cell if 3 neighbor cells are live
    
    return U_new


def main( U0 , NT ):
    
    U = []
    U.append( U0 )
    
    for i in range(NT):
        
        U.append( gol_step( U[-1] ) ) # appending the last output of the 32x32 grid into list
    
    return U
        

    
if __name__ == '__main__':
    
    N   = 32
    NT  = 128
    U0  = np.array( [ [ np.random.choice([0,1]) for i in range(N) ] for j in range(N) ] ) # list comprehension!
    
    U   = main( U0 , NT )
    
    plt.ion()
    
    for i in range(len(U)):
        
        plt.imshow( U[i] )
        plt.title( "Conway's Game of Life " + str(i+1) )
        plt.draw()
        plt.pause(0.1)
        plt.clf()
        
