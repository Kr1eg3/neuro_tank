from easyvec import Vec2, Rect
from easyvec.geometry import PolyLine, intersect, is_in_polygon, _sortreduce_by_distance
import numpy as np


def rnd_vec(mx, my, dx, dy):
    x = np.random.uniform(mx-dx, mx+dx)
    y = np.random.uniform(my-dy, my+dy)
    return Vec2(x, y)

def rnd_polygon(x, y, r, n, dx, dy, angle0=None):
    v = Vec2(r, 0)
    v.rotate_( angle0 if angle0 else np.random.uniform(0,360) , degrees=True)
    vs = [v.rotate(i * (360/n), degrees=True) + rnd_vec(x, y, dx, dy) 
        for i in range(n)]
    return PolyLine(vs, enclosed=True)

def get_obstacles(w, h):
    r1 = Rect(0,0, w/4, h/4)
    r2 = Rect(w,h, w-w/4, h-h/4)
    

class Scene(object):
    @classmethod
    def get_standart(cls, w: float, h: float, n: int, m: int):
        obs = [PolyLine([
            Vec2(0,0), Vec2(w,0), Vec2(w,h), Vec2(0, h)
        ], enclosed=True),
        PolyLine([
            Vec2(-10,-10), Vec2(w+10,-10), Vec2(10+w,10+h), Vec2(-10, 10+h)
        ], enclosed=True)]
        w1, h1 = w / n, h / m
        r = min(w1, h1) * 0.3
        for i in range(n):
            for j in range(m):
                x = w1/2 + w1 * i + np.random.uniform(-0.1, 0.1) * w1
                y = h1/2 + h1 * j + np.random.uniform(-0.1, 0.1) * h1
                if (i==0 and j==0): 
                    pos_1 = Vec2(x, y)
                    continue
                if (i==(n-1) and j==(m-1)):
                    pos_2 = Vec2(x, y)
                    continue
               
                ri = r * np.random.uniform(0.9, 1.1)
                nn = np.random.randint(3,6)
                dx = w1 * np.random.uniform(0.02, 0.1)
                dy = h1 * np.random.uniform(0.02, 0.1)
                obs.append(rnd_polygon(x,y,ri, nn, dx, dy))
        return cls(obs, pos_1, pos_2)



    def __init__(self, obstacles, pos_1=None, pos_2=None):
        self.obstacles = obstacles
        self.pos_1 = pos_1
        self.pos_2 = pos_2

    def get_vis_polygon(self, pos: Vec2, r: float, alpha: float, thetta: float) -> PolyLine:
        n_rays = 10
        angles = np.linspace(-thetta+alpha, thetta+alpha, n_rays)
        p2s = [Vec2(r ,0).rotate(angle, degrees=True).add(pos) for angle in angles]
        
        ps = [pos.copy()]
        for p2 in p2s:
            p1, pi = self.intersected_segment(pos, p2)
            ps.append(pi)
        return PolyLine(ps)

    def intersected_segment(self, p1: Vec2, p2: Vec2) -> tuple:
        ps = [p2.copy()]
        for pl in self.obstacles:
            ps += pl.intersect_segment(p1, p2)
        ps = _sortreduce_by_distance(ps, p1)
        return (p1.copy(), ps[0])



if __name__ == "__main__":
    pl = rnd_polygon(200, 100, 50, 3, 20, 20)
    print(pl.is_in(Vec2(200,100)))
    
