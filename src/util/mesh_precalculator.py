import os
import numpy as np
from tqdm import tqdm


def make_mat(n: int, m: int):
    mat = np.arange(0, (n * m), 1).reshape(n, m)
    circular_indies = [0, 1, 3, 2]
    faces = []
    for _n in range(n - 1):
        for _m in range(m - 1):
            face = mat[_n : _n + 2 : 1, _m : _m + 2 : 1].flatten().tolist()
            faces.append(" ".join(str(face[index]) for index in circular_indies))

    with open(
        "C:\\Users\\skyfe\\Development\\diffconvpl\\data\\generated\\precalculated\\face_spacing_0_02.txt",
        "r",
    ) as bench:
        for i, line in enumerate(bench.readlines()):
            assert (
                line.strip() == faces[i]
            ), f"Line {i} does not match: '{line.strip()}' --> '{faces[i]}'"


if __name__ == "__main__":
    n = 101
    m = 101
    make_mat(n, m)
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
