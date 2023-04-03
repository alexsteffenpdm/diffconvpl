import numpy as np

class GridExporter():
    def __init__(self,gridshape:np.shape):
        print(gridshape)
        self.n = gridshape[0]
        self.m = gridshape[1]
        self.grid:list[str] = self.make_grid()

    def make_grid(self,) -> list[str]:
        mat = np.arange(0,(self.n*self.m),1).reshape(self.n,self.m)
        circular_indies = [0,1,3,2]
        faces = []    
        for _n in range(self.n-1):
            for _m in range(self.m-1):
                face = mat[_n:_n+2:1,_m:_m+2:1].flatten().tolist()
                faces.append(" ".join(str(face[index]) for index in circular_indies))
        return faces

if __name__ == "__main__":
    n,m = 201,201
    arr = np.zeros((n,m))
    exporter = GridExporter(arr.shape)
    print(exporter.grid)
