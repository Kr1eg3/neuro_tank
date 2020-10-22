from easyvec import Vec2
from easyvec.geometry import PolyLine, _convert
from unit import Unit, Round

class Screen(object):
    """Класс для отображения состояния имитационной модели на плоскости. Требует библиотеки pygame ($pip install pygame)
    """
    def __init__(self, display_width, display_height, v0, v1, clock_tick=120):
        """Конструктор
        
        Arguments:
            display_width {int} -- ширина окна отображения в пикселях
            display_height {int} -- высота окна отображения в пикселях
            v0 {float} -- координаты нижней левой координаты оботбражения
            v1 {float} -- координаты верхней правой координаты оботбражения
            clock_tick
        """
        self.v1 = _convert(v1)
        self.v0 = _convert(v0)
        self.display_height = display_height
        self.display_width = display_width
        self.pygame = None
        self.clock_tick = clock_tick

    def to_pixels(self, vec):
        """Метод перевода координат имитационной модели в пиксели на экране отображения
        
        Arguments:
            vec {Vec2d} -- координата для перевода
        
        Returns:
            tuple(int, int) -- пиксельные координаты, соответствующие координате для перевода
        """
        masht_x = self.display_width / (self.v1.x - self.v0.x)
        masht_y = self.display_height / (self.v1.y - self.v0.y)

        x = int((vec.x - self.v0.x)*masht_x)
        y = int((self.v1.y - vec.y)*masht_y)
        return x, y

    def to_vec(self, x, y):
        """Метод перевода пиксельных координат в координаты имитационной модели
        
        Arguments:
            x {int} -- пиксельная координата по оси абсцисс
            y {int} -- пиксельная координата по оси ординат
        
        Returns:
            Vec2d -- соответствующая координата имитационной модели
        """
        masht_x = self.display_width / (self.v1.x - self.v0.x)
        masht_y = self.display_height / (self.v1.y - self.v0.y)

        res = Vec2(self.v0.x + x/masht_x, self.v1.y - y/masht_y)
        return res

    def update(self, clock_tick=None):
        """Обновить экран отображения
        
        Keyword Arguments:
            clock_tick {int} -- максимальное количество кадров в секунду (default: {None})
        """
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.pygame.quit()
                quit()
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_SPACE:
                    print('PAUSED')
                    paused = True
                    while paused:
                        self.pygame.display.update()
                        if clock_tick is not None:
                            self.clock.tick(clock_tick)
                        else:
                            self.clock.tick(self.clock_tick)
                        for event in self.pygame.event.get():
                            if event.type == self.pygame.QUIT:
                                self.pygame.quit()
                                quit()
                            elif event.type == self.pygame.KEYDOWN:
                                if event.key == self.pygame.K_SPACE:
                                    paused = False
                                    print('UNPAUSED')

        self.pygame.display.update()
        if clock_tick is not None:
            self.clock.tick(clock_tick)
        else:
            self.clock.tick(self.clock_tick)

    def draw(self, who):
        self._initpygame()
        if isinstance(who, PolyLine):
            self.draw_poly_line(who)
        elif isinstance(who, Unit):
            self.draw_unit(who)
        elif isinstance(who, Round):
            self.draw_round(who)
        elif isinstance(who, str):
            self.draw_text(who)

    def draw_text(self, t: str):
        text = self.myfont.render(t, False, (0,0,0))
        self.screen_main.blit(text, (0,0))

    def draw_round(self, r: Round, color=None):
        if color is None:
            color = (255,0,0)
        self.pygame.draw.circle(
            self.screen_main,
            color,
            self.to_pixels(r.pos),
            3
        )

    def draw_vis_poly(self, u: Unit):
        if u.vis_polygon and u.draw_stuff:
            self.pygame.draw.polygon(
                self.screen_main, 
                (255,255,200),
                [self.to_pixels(p) for p in u.vis_polygon.vecs]
            )

    def draw_unit(self, u: Unit):
        if u.draw_stuff:
            rays = u.intersected if u.intersected else u.get_rays_world()
            for p1, p2 in rays:
                self.pygame.draw.aaline(
                    self.screen_main, 
                    (220,220,220),
                    self.to_pixels(p1),
                    self.to_pixels(p2)
                )
            if u.vis_line:
                self.pygame.draw.aaline(
                    self.screen_main, 
                    (255,50,50),
                    self.to_pixels(u.pos),
                    self.to_pixels(u.vis_line + u.pos)
                )

        self.pygame.draw.polygon(
            self.screen_main, 
            u.color,
            [self.to_pixels(p) for p in u.polygon_world.vecs]
            )


        hp_w = 30
        hp_h = 5
        t = u.hp/u.hp0
        x, y = self.to_pixels(u.pos)
        rr = self.pygame.Rect(x - 15, y - 20, hp_w, hp_h)
        rg = self.pygame.Rect(x - 15, y - 20, int(hp_w*t), hp_h)
        self.pygame.draw.rect(self.screen_main, (255,0,0), rr)
        self.pygame.draw.rect(self.screen_main, (0,255,0), rg)

        if not u.can_shoot:
            t2 = (u.time - u.time_last_shot) / u.shot_delta
            rr = self.pygame.Rect(x - 15, y - 15, int(hp_w*t2), 2)
            self.pygame.draw.rect(self.screen_main, (150,150,150), rr)



    def draw_poly_line(self, pl: PolyLine, color=None):
        if color is None:
            color = (0,0,0)
        self.pygame.draw.aalines(
            self.screen_main, 
            color, 
            pl.enclosed, 
            [self.to_pixels(v) for v in pl.vecs], 
            2)

    def fill(self, color=None):
        """Заполнить экран сплошным цветом
        
        Keyword Arguments:
            color {tuple} -- код цвета (по-умолчанию белый (255,255,255)) (default: {None})
        """
        self._initpygame()
        self.screen_main.fill(color if color else (255, 255, 255))

    def close(self):
        """Закончить отображение
        """
        if self.pygame:
            self.pygame.quit()

    def __del__(self):
        """Destructor
        """
        self.close()

    def _initpygame(self):
        if self.pygame:
            return
        import pygame
        self.pygame = pygame
        self.pygame.init()
        self.pygame.font.init()
        self.screen_main = pygame.display.set_mode((self.display_width, self.display_height))   
        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont('Arial', 30)

if __name__ == "__main__":
    from scene import Scene
    scr = Screen(700,700,(-10,-10), (60, 60))
    sc = Scene.get_standart(50, 50, 3 , 3)

    u = Unit.get_some((10,10))

    scr.fill()
    for pl in sc.obstacles:
        scr.draw(pl) 
    scr.draw(u)
    scr.update()
    input()