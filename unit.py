from easyvec import Vec2, Mat2, PolyLine
from easyvec.geometry import _convert
from math import pi, sqrt
import numpy as np

def random_behaivor(*args, **kwargs):
    def inner(d, name, v1, v2):
        if np.random.random() < 0.3:
            d[name] = np.random.uniform(v1, v2)
    res = {}
    inner(res, 'move', 0.9, 1)
    inner(res, 'rotate', -0.5, 1)
    inner(res, 'vision', -1, 0.8)
    inner(res, 'fire', 0.5, 1)
    return res    

class Unit(object):
    @classmethod
    def get_some(cls, pos, uname='test_unit', color=(30,255,30), brain_foo=None):
        bf = brain_foo if brain_foo else random_behaivor
        return cls(
            name=uname, 
            pos=pos, 
            alpha=np.random.uniform(-170,170), 
            brain_foo=bf, 
            theta=90, 
            r_vis=15, 
            v_max=1, 
            d_alpha_max=20, 
            n_rays=10, 
            dmg=1, 
            round_vel=2, 
            hp=3, 
            shot_delta=5, 
            d_theta_max=20,
            color=color)

    def __init__(self, name, pos, alpha, brain_foo, theta, r_vis, v_max, d_alpha_max, n_rays, dmg, round_vel, 
            hp, shot_delta, d_theta_max, color, draw_stuff=True):
        self.name = name
        self.pos = _convert(pos)
        self.alpha = alpha
        self.brain_foo = brain_foo
        self.theta = theta
        self.d_theta_max = d_theta_max
        self.r_vis = r_vis
        self.v_max = v_max
        self.d_alpha_max = d_alpha_max
        self.n_rays = n_rays
        self.s_vis = pi * r_vis**2 / 2
        self.polygon_local = PolyLine([Vec2(2,0), Vec2(-1,-1), Vec2(-1,1)], enclosed=True)
        self.dmg = dmg
        self.round_vel = round_vel
        self.hp = hp
        self.hp0 = hp
        self.time = 0
        self.time_last_shot = -shot_delta
        self.shot_delta = shot_delta

        self.vis_polygon = None
        self.vis_line = None
        self.intersected = None

        self.set_alpha(alpha)
        self.set_theta(theta)
        self.color = color
        self.draw_stuff = draw_stuff

    @property
    def can_shoot(self):
        return self.time - self.shot_delta >= self.time_last_shot

    def get_move_segment(self, vel_portion: float, dalpha_portion: float, dt: float):
        if abs(vel_portion) < 1e-8:
            return None
        vel_portion = 1 if vel_portion > 1 else -1 if vel_portion < -1 else vel_portion
        dalpha_portion = 1 if dalpha_portion > 1 else -1 if dalpha_portion < -1 else dalpha_portion
        vel = Vec2(self.v_max*vel_portion * dt, 0).rotate(self.alpha + dalpha_portion*self.d_alpha_max*dt/2, degrees=True)
        return (self.pos.copy(), self.pos + vel)

    def rotate(self, rotate_signal, dt):
        self.set_alpha(self.alpha + dt * rotate_signal * self.d_alpha_max)

    def change_vision(self, d_vision, dt):
        thetta_new = self.theta + dt * d_vision * self.d_theta_max
        thetta_new = 180 if thetta_new > 180 else 2 if thetta_new < 2 else thetta_new
        self.set_theta(thetta_new)

    @property
    def polygon_world(self) -> PolyLine:
        pl1 = self.polygon_local.transform(self.M_rot_1)
        pl2 = pl1.add_vec(self.pos)
        return pl2

    def set_theta(self, theta):
        self.theta = theta
        self.r_vis = sqrt(self.s_vis/theta * 180 / pi)
        self.angles = np.linspace(-self.theta, self.theta, self.n_rays)
        self.rays = [Vec2(self.r_vis ,0).rotate(angle, degrees=True) for angle in self.angles]

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.M_rot = Mat2.from_angle(alpha, degrees=True)
        self.M_rot_1 = self.M_rot._1

    def to_world(self, local: Vec2, is_dir_vector=False) -> Vec2:
        if is_dir_vector:
            return self.M_rot_1 * local
        return self.M_rot_1 * local + self.pos

    def to_local(self, world: Vec2, is_dir_vector=False) -> Vec2:
        if is_dir_vector:
            return self.M_rot * world
        return self.M_rot * (world - self.pos)

    def get_rays_world(self):
        return [(self.pos.copy(), self.to_world(ray_p)) for ray_p in self.rays]        

    def get_comand(self, signals: dict):
        if self.brain_foo:
            res = self.brain_foo(signals)
            if isinstance(res, dict):
                return res
        return {}

    def shoot(self):
        if not self.can_shoot:
            return None
        angle = np.random.normal(self.alpha, self.theta/3)
        vel = Vec2(self.round_vel, 0).rotate(angle, degrees=True)
        self.time_last_shot = self.time
        v = 3*vel.norm() + self.pos
        return Round(v, vel, self.dmg)

    



class Round(object):
    id_counter = 0

    def __init__(self, pos, vel, dmg):
        self.pos = _convert(pos).copy()
        self.vel = _convert(vel).copy()
        self.dmg = dmg
        self.id = Round.id_counter
        Round.id_counter += 1

    def get_move_segment(self, dt: float):
        return (self.pos.copy(), self.pos + dt * self.vel)



if __name__ == "__main__":
    m = Mat2.from_angle(90, True)
    print(m._1 * Vec2(1,0))

    