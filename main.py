print("Dijkistra's + backtracking  Path finding in maze")
print("Select a option")
print("1. Solve maze by taking input as image")
a = input()

if a == '1':
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    # read the image
    img = cv2.imread('download.png')
    plt.figure(figsize=(7, 7))
    plt.imshow(img)


    # Specifying starting and ending point
    cv2.circle(img, (50, 20), 3, (255, 0, 0), -1)
    cv2.circle(img, (30,370), 3, (0, 0, 255), -1)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.show()


    class Vertex:
        def __init__(self, x_coord, y_coord):
            self.x = x_coord
            self.y = y_coord
            self.d = float('inf')  # distance from source
            self.parent_x = None
            self.parent_y = None
            self.processed = False
            self.index_in_queue = None


    # Returns neighbor
    def get_neighbors(mat, r, c):
        shape = mat.shape
        neighbors = []
        if r > 0 and not mat[r - 1][c].processed:
            neighbors.append(mat[r - 1][c])
        if r < shape[0] - 1 and not mat[r + 1][c].processed:
            neighbors.append(mat[r + 1][c])
        if c > 0 and not mat[r][c - 1].processed:
            neighbors.append(mat[r][c - 1])
        if c < shape[1] - 1 and not mat[r][c + 1].processed:
            neighbors.append(mat[r][c + 1])
        return neighbors


    # calculating Euclidean distance
    def get_distance(img, u, v):
        return 0.1 + (float(img[v][0]) - float(img[u][0])) ** 2 + (float(img[v][1]) - float(img[u][1])) ** 2 + (
                float(img[v][2]) - float(img[u][2])) ** 2


    def bubble_up(queue, index):
        if index <= 0:
            return queue
        p_index = (index - 1) // 2
        if queue[index].d < queue[p_index].d:
            queue[index], queue[p_index] = queue[p_index], queue[index]
            queue[index].index_in_queue = index
            queue[p_index].index_in_queue = p_index
            queue = bubble_up(queue, p_index)
        return queue


    def bubble_down(queue, index):
        length = len(queue)
        lc_index = 2 * index + 1
        rc_index = lc_index + 1

        if lc_index >= length:
            return queue

        if lc_index < length and rc_index >= length:

            if queue[index].d > queue[lc_index].d:
                queue[index], queue[lc_index] = queue[lc_index], queue[index]
                queue[index].index_in_queue = index
                queue[lc_index].index_in_queue = lc_index
                queue = bubble_down(queue, lc_index)

        else:
            small = lc_index
            if queue[lc_index].d > queue[rc_index].d:
                small = rc_index
            if queue[small].d < queue[index].d:
                queue[index], queue[small] = queue[small], queue[index]
                queue[index].index_in_queue = index
                queue[small].index_in_queue = small
                queue = bubble_down(queue, small)

        return queue


    def find_shortest_path(img, src, dst):
        pq = []  # min-heap priority queue

        source_x = src[0]
        source_y = src[1]
        dest_x = dst[0]
        dest_y = dst[1]

        imagerows, imagecols = img.shape[0], img.shape[1]
        matrix = np.full((imagerows, imagecols), None)

        for r in range(imagerows):
            for c in range(imagecols):
                matrix[r][c] = Vertex(c, r)
                matrix[r][c].index_in_queue = len(pq)
                pq.append(matrix[r][c])

        matrix[source_y][source_x].d = 0

        pq = bubble_up(pq, matrix[source_y][source_x].index_in_queue)

        while len(pq) > 0:

            u = pq[0]
            u.processed = True
            pq[0] = pq[-1]
            pq[0].index_in_queue = 0
            pq.pop()
            pq = bubble_down(pq, 0)

            neighbors = get_neighbors(matrix, u.y, u.x)

            for v in neighbors:
                dist = get_distance(img, (u.y, u.x), (v.y, v.x))
                if u.d + dist < v.d:
                    v.d = u.d + dist
                    v.parent_x = u.x
                    v.parent_y = u.y
                    idx = v.index_in_queue
                    pq = bubble_down(pq, idx)
                    pq = bubble_up(pq, idx)

        path = []
        iter_v = matrix[dest_y][dest_x]
        path.append((dest_x, dest_y))

        while (iter_v.y != source_y or iter_v.x != source_x):
            path.append((iter_v.x, iter_v.y))
            iter_v = matrix[iter_v.parent_y][iter_v.parent_x]

        path.append((source_x, source_y))
        return path


    def drawPath(img, path, thickness=2):
        x0, y0 = path[0]

        for vertex in path[1:]:
            x1, y1 = vertex
            cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), thickness)
            x0, y0 = vertex


    # Calling shortest path Function
    cv2.imwrite('maze-initial.png', img)

    # calling shortest path function
    p = find_shortest_path(img, (50, 20), (30, 370))

    # calling drawing path
    drawPath(img, p)

    cv2.imwrite('maze-solution.png', img)
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.show()

########################################################

# maze generation :
""" # Maze generation using recursive backtracker algorithm
# https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_backtracker

from queue import LifoQueue
from PIL import Image
import numpy as np
from random import choice

def generate_maze(width, height):
    maze = Image.new('RGB', (2*width + 1, 2*height + 1), 'black')
    pixels = maze.load()

    # Create a path on the very top and bottom so that it has an entrance/exit
    pixels[1, 0] = (255, 255, 255)
    pixels[-2, -1] = (255, 255, 255)

    stack = LifoQueue()
    cells = np.zeros((width, height))
    cells[0, 0] = 1
    stack.put((0, 0))

    while not stack.empty():
        x, y = stack.get()

        adjacents = []
        if x > 0 and cells[x - 1, y] == 0:
            adjacents.append((x - 1, y))
        if x < width - 1 and cells[x + 1, y] == 0:
            adjacents.append((x + 1, y))
        if y > 0 and cells[x, y - 1] == 0:
            adjacents.append((x, y - 1))
        if y < height - 1 and cells[x, y + 1] == 0:
            adjacents.append((x, y + 1))

        if adjacents:
            stack.put((x, y))

            neighbour = choice(adjacents)
            neighbour_on_img = (neighbour[0]*2 + 1, neighbour[1]*2 + 1)
            current_on_img = (x*2 + 1, y*2 + 1)
            wall_to_remove = (neighbour[0] + x + 1, neighbour[1] + y + 1)

            pixels[neighbour_on_img] = (255, 255, 255)
            pixels[current_on_img] = (255, 255, 255)
            pixels[wall_to_remove] = (255, 255, 255)

            cells[neighbour] = 1
            stack.put(neighbour)

    return maze


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("width", nargs="?", type=int, default=32)
    parser.add_argument("height", nargs="?", type=int, default=None)
    parser.add_argument('--output', '-o', nargs='?', type=str, default='generated_maze.png')
    args = parser.parse_args()

    size = (args.width, args.height) if args.height else (args.width, args.width)

    maze = generate_maze(*size)
"""
