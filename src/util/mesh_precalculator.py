import os
import numpy as np
from tqdm import tqdm
SPACINGS = [
    0.75,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    0.33,
    0.5,
    0.66,
    0.8,
    0.99,
    1.0,
]
MIN = -1.0
MAX = 1.0

FILENAME_TEMPLATE = "face_spacing_"
DUMMY_FUNC = lambda x,y: 0

def create_faces(edges:np.array,dimensions:np.shape):
    faces = []                                                                                                                                                                                                                                                                                                                                                                                                                                 
    dp = np.arange(0,dimensions[0]*dimensions[1])
    print("Generating Faces")
    for x in tqdm(dp):
        numbers = [x,x+1,x+(dimensions[1]+1),x+(dimensions[1])]
        _edges = [[x,x+1], [x,x+(dimensions[1])],[x+1,x+(dimensions[1]+1)],[x+(dimensions[1]),x+(dimensions[1])+1]]
        IN_EDGES = []
        for e in _edges:
            add = False
            for E in edges:
                if e[0] == E[0] and e[1] == E[1]:
                    add = True
                    break
            IN_EDGES.append(add)

        if IN_EDGES.count(False) == 0:
            faces.append(numbers)            
    print(f"Generated {len(faces)} faces")
    return faces

def create_edges(arr:np.array):
    dp = np.arange(0,arr.shape[0]*arr.shape[1],1).reshape(arr.shape)
    edges = []
    print("Generating edges")
    for x in tqdm(range(arr.shape[1])):
        for y in range(arr.shape[0]):
            _e = []
            if x+1 < arr.shape[1]:
                _e.append([dp[y][x],dp[y][x+1]])
            if x-1 > 0:
                _e.append([dp[y][x],dp[y][x-1]])
            if y-1 > 0:
                _e.append([dp[y][x],dp[y-1][x]])
            if y+1 < arr.shape[0]:
                _e.append([dp[y][x],dp[y+1][x]])

            for e in _e:
                edges.append(sorted(e))

    return np.asanyarray(list(set(tuple(sorted(sub)) for sub in edges)))

def precalculate(spacing:float=None):
    if not os.path.exists(os.path.join(os.getcwd(),"data","generated","precalculated")):
        os.makedirs(os.path.join(os.getcwd(),"data","generated","precalculated"))
    _SPACING = []
    if spacing is not None:
        _SPACING.append(spacing)
    else:
        _SPACING = SPACINGS
    for spacing in _SPACING:
        points:np.array = int((abs(MIN)+abs(MAX)) / spacing) + 1
        domain: np.array = np.linspace(MIN,MAX,points)
        x,y = np.meshgrid(domain,domain)        
        edges = create_edges(x)
        faces = create_faces(edges,x.shape)

        with open(
            os.path.join(
            os.getcwd(),"data","generated","precalculated",f"{FILENAME_TEMPLATE}{str(spacing).replace('.','_')}.txt"),
            "w"
            ) as file:
                  for i,f in enumerate(faces):
                    if i != len(faces)-1:
                        file.write(f"{f[0]} {f[1]} {f[2]} {f[3]}\n")
                    else:
                        file.write(f"{f[0]} {f[1]} {f[2]} {f[3]}")





def make_mat(n:int,m:int):
  
    mat = np.arange(0,(n*m),1).reshape(n,m)
    circular_indies = [0,1,3,2]
    faces = []    
    for _n in range(n-1):
        for _m in range(m-1):
            face = mat[_n:_n+2:1,_m:_m+2:1].flatten().tolist()
            faces.append(" ".join(str(face[index]) for index in circular_indies))

    with open("C:\\Users\\skyfe\\Development\\diffconvpl\\data\\generated\\precalculated\\face_spacing_0_02.txt","r") as bench:
        for i,line in enumerate(bench.readlines()):
            assert line.strip() == faces[i], f"Line {i} does not match: '{line.strip()}' --> '{faces[i]}'"

    




 

if __name__ == "__main__":
    n = 101
    m = 101
    make_mat(n,m)
    # if not os.path.exists(os.path.join(os.getcwd(),"data","generated","precalculated")):
    #     os.makedirs(os.path.join(os.getcwd(),"data","generated","precalculated"))

    # for spacing in SPACINGS:
    #     points:np.array = int((abs(MIN)+abs(MAX)) / spacing) + 1
    #     domain: np.array = np.linspace(MIN,MAX,points)
    #     x,y = np.meshgrid(domain,domain)
    #     x_f = x.flatten()
    #     y_f = y.flatten()
    #     z = np.zeros_like(x.flatten())
    #     verts = np.asanyarray([[xi,yi,zi] for xi,yi,zi in zip(x_f,y_f,z)])
    #     edges = create_edges(x)
    #     faces = create_faces(edges,x.shape)

    #     with open(
    #         os.path.join(
    #         os.getcwd(),"data","generated","precalculated",f"{FILENAME_TEMPLATE}{str(spacing).replace('.','_')}.txt"),
    #         "w"
    #         ) as file:
    #               for i,f in enumerate(faces):
    #                 if i != len(faces)-1:
    #                     file.write(f"{f[0]} {f[1]} {f[2]} {f[3]}\n")
    #                 else:
    #                     file.write(f"{f[0]} {f[1]} {f[2]} {f[3]}")
        
