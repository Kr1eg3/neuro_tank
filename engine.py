from unit import Unit, random_behaivor
from screen import Screen
from scene import Scene
from easyvec import Vec2
import numpy as np

class Engine(object):
    def __init__(self, scene: Scene, screen: Screen, tank: Unit, enemy: Unit, **kwargs):
        self.scene = scene
        self.screen = screen
        self.units = [tank, enemy]
        self.rounds = set()
        self.time = 0
        self.dt = kwargs.get('dt', 0.1)
        self.step_count = 0

    def render(self):
        if self.screen is None:
            return
        scr = self.screen
        scr.fill()
        for u in self.units:
            scr.draw_vis_poly(u)
        for pl in self.scene.obstacles:
            scr.draw(pl)
        for r in self.rounds:
            scr.draw(r)
        for u in self.units:
            scr.draw(u)
        scr.draw(f'steps = {self.step_count}, time = {self.time:.1f}')
        scr.update()
        # if self.step_count % 4 == 0:
        #     scr.pygame.image.save(scr.screen_main ,f'tmp/{self.step_count}.jpg')
        

    def step(self, render=True):       
        log = []
        result = {
            'log': log,
            'time': self.time,
            'step_count': self.step_count}
        signals = [self.get_signals(u) for u in self.units]
        comands = [u.get_comand(s) for u, s in zip(self.units, signals)]
        result['signals_comands'] = list(zip(signals, comands))
        if self.step_count == 0 and render:
            self.render()
        for u, c in zip(self.units, comands):
            self.unit_step(u, c, self.dt, log)
        self.time += self.dt
        self.rounds_steps(self.dt, log)
        self.step_count += 1
        if render:
            self.render()
        return result

    @property
    def done(self):
        if self.step_count >= 70000:
            return True
        for u in self.units:
            if u.hp <= 0:
                return True
        return False 

    def get_result(self):
        u1 = self.units[0]
        u2 = self.units[1]
        if abs(u1.hp - u2.hp) < 1e-8:
            return -1.0
        #return f'Победил игрок {u1.name if u1.hp > u2.hp else u2.name}'
        elif u1.hp > u2.hp:
            return -1.0
        else:
            return 1.0

    def rounds_steps(self, dt: float, log: list):
        neet2explode = set()
        u_polygons = [(u.polygon_world, u) for u in self.units]
        for r in self.rounds:
            p1, p2 = r.get_move_segment(dt)
            p1, p3 = self.scene.intersected_segment(p1, p2)
            if p3 != p2:
                neet2explode.add(r)
                p2 = p3
                log.append(f'Round id={r.id} hitted wall')
            for up, u in u_polygons:
                ps = up.intersect_segment(p1, p2)
                if ps:
                    neet2explode.add(r)
                    p2 = ps[0]
                    log.append(f'Unit name={u.name} hitted by Round id={r.id} by {r.dmg}')
                    hp0 = u.hp
                    u.hp -= r.dmg
                    hp1 = u.hp
                    log.append(f'Unit name={u.name} changed hp by from {hp0} to {hp1}')
            r.pos = p2
            log.append(f'Round id={r.id} moved from {p1} to {p2}')

        for r in neet2explode:
            self.rounds.discard(r)
            log.append(f'Round id={r.id} deleted')
        

    def _validate(self, name: str, comands: dict,  vmin, vmax):
        try:
            v = comands.get(name, 0)
            v = vmax if v > vmax else vmin if v < vmin else v
            return v
        except:
            return 0

    def unit_step(self, unit: Unit, comands: dict, dt: float, log: list):
        move = self._validate('move', comands, -1, 1)
        rotate = self._validate('rotate', comands, -1, 1)
        vision = self._validate('vision', comands, -1, 1)
        fire = self._validate('fire', comands, 0, 1)
        if abs(move) > 1e-8:
            p1, p2 = unit.get_move_segment(move, rotate, dt)
            p1, p3 = self.scene.intersected_segment(p1, p2)
            if p3 != p2:
                unit.pos = p1
            else:
                unit.pos = p2
                log.append(f'Unit name={unit.name} moved from {p1} to {unit.pos}')
        
        if abs(rotate) > 1e-8:
            a0 = unit.alpha
            unit.rotate(rotate, dt)
            a1 = unit.alpha
            log.append(f'Unit name={unit.name} rotated from {a0} to {a1}')

        if abs(vision) > 1e-8:
            t0, r0 = unit.theta, unit.r_vis
            unit.change_vision(vision, dt)
            t1, r1 = unit.theta, unit.r_vis
            log.append(f'Unit name={unit.name} change vision theta from {t0} to {t1}')
            log.append(f'Unit name={unit.name} change vision radius from {r0} to {r1}')

        if fire > 1e-8:
            round_new = unit.shoot()
            if round_new:
                self.rounds.add(round_new)
                log.append(f'Unit name={unit.name} fired by Round id={round_new.id} with pos={round_new.pos}, vel={round_new.vel}')
        unit.time += dt



    def get_signals(self, unit: Unit):
        res = {}
        self.get_info_signals(unit, res)
        self.add_ray_signals(unit, res)
        self.add_pos_signals(unit, res)
        self.add_vis_signals(unit, res)
        return res

    def get_info_signals(self, unit: Unit, signals: dict):
        signals['name'] = unit.name
        signals['time'] = self.time
        signals['step_count'] = self.step_count
        signals['hp'] = unit.hp
        signals['can_shoot'] = unit.can_shoot

    def add_vis_signals(self, unit: Unit, signals: dict):
        unit.vis_polygon = self.scene.get_vis_polygon(unit.pos, unit.r_vis, unit.alpha, unit.theta)
        signals['enemies'] = []
        signals['rounds'] = []
        for u in self.units:
            if u is not unit:
                upos = u.pos
                if unit.vis_polygon.is_in(upos):
                    unit.vis_line = upos - unit.pos
                    unit_ox = unit.to_world(Vec2(1,0), is_dir_vector=True)
                    angle = unit_ox.angle_to(unit.vis_line, degrees=True)
                    signals['enemies'].append((angle, unit.vis_line.len()))
                else:
                    unit.vis_line = None

        for u in self.rounds:
            upos = u.pos
            if unit.vis_polygon.is_in(upos):
                vis_line = upos - unit.pos
                unit_ox = unit.to_world(Vec2(1,0), is_dir_vector=True)
                angle = unit_ox.angle_to(vis_line, degrees=True)
                signals['rounds'].append((angle, vis_line.len()))


    def add_pos_signals(self, unit: Unit, signals: dict):
        # signals['pos'] = unit.pos.as_tuple()
        signals['alpha'] = unit.alpha
        signals['theta'] = unit.theta
        signals['r_vis'] = unit.r_vis

    def add_ray_signals(self, unit: Unit, signals: dict):
        rays = unit.get_rays_world()
        unit.intersected = [self.scene.intersected_segment(p1, p2) for p1, p2 in rays]
        # signals['rays_max'] = [(p1-p2).len() for p1, p2 in rays],
        signals['rays_intersected'] = [(p1-p2).len() for p1, p2 in unit.intersected]
        

    @classmethod
    def get_standart(cls, you_brain_foo, enemy_brain_foo=None):
        scr = Screen(700,700,(-10,-10), (60, 60))
        sc = Scene.get_standart(50, 50, 3 , 3)
        poss = [sc.pos_1, sc.pos_2]
        np.random.shuffle(poss)
        p1, p2 = poss
        u1 = Unit.get_some(p1, 'you', color=(36,235,130), brain_foo=you_brain_foo)
        u2 = Unit.get_some(p2, 'enemy',color=(233,44,44), brain_foo=enemy_brain_foo)
        u2.draw_stuff = False
        return cls(sc, scr, u2, u1)

    @classmethod
    def get_standart2(cls, you_brain_foo, enemy_brain_foo=None):
        scr = Screen(700,700,(-10,-10), (60, 60))
        sc = Scene.get_standart(50, 50, 3 , 3)
        p1, p2 = sc.pos_1, sc.pos_1 + (10, 10)
        u1 = Unit.get_some(p1, 'you', color=(36,235,130), brain_foo=you_brain_foo)
        u2 = Unit.get_some(p2, 'enemy',color=(233,44,44), brain_foo=enemy_brain_foo)
        u1.set_alpha(30)
        u2.set_alpha(170)
        u2.draw_stuff = False
        res = cls(sc, scr, u2, u1) 
        # res.units = []
        return res

if __name__ == "__main__":
    from pprint import pprint
    # scr = Screen(700,700,(-10,-10), (60, 60))
    # sc = Scene.get_standart(50, 50, 3 , 3)
    # u1 = Unit.get_some((10,10), 'unit1', color=(0,0,255))
    # u2 = Unit.get_some((15,15), 'unit2')
    eng = Engine.get_standart2(None)
    import time

    while not eng.done:
        info = eng.step(render=True)
        # print(info['step_count'])
        # pprint(info)
        log = info['log']
        for mess in log:
            if 'hp' in mess:
                print(mess)
    
    print(eng.get_result())