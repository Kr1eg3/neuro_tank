import numpy as np


class MyObject:
    def __init__(self, fullhp: int, nrays: int, dt: float = .1):
        self.acc = 5        # Для сравнения float
        self.switch = 0

        self.nrays = nrays
        self.midray = nrays // 2
        self.max_theta = 90

        self.step = None
        self.alpha = None
        self.shooting = False
        self.enemies = None
        self.hp = None
        self.r_vis = None
        self.rays_intersected = None
        self.rounds = None
        self.theta = None
        self.time = None

        self.behaivor = {}
        self.min_range = 5                                  # Минимальное расстояние до препятствий (для макс. луча)
        self.alpha_to_enemy = None                          # alpha на цель
        self.is_contact, self.was_contact = False, False    # Контакт сейчас, контакт был

        self.steps = 0                                      # Участвует в том, чтобы поиск цели не был вечным
        self.lim_steps = 50

        self.self_damage = False
        self.hp_buf = fullhp

        self.rollback = False
        self.max_stack = 300
        self.traj_stack = []

        self.sum_angle = 0

        self.dt = dt
        self.timer = 0

    def update_info(self, data: dict):
        # Сброс словаря
        self.behaivor['fire'] = 0
        self.behaivor['move'] = 0
        self.behaivor['rotate'] = 0
        self.behaivor['vision'] = 0
        # Перезапись данных
        self.step = data['step_count']
        self.alpha = data['alpha']
        self.shooting = data['can_shoot']
        self.enemies = data['enemies']
        self.hp = data['hp']
        self.r_vis = data['r_vis']
        self.rays_intersected = np.array(data['rays_intersected'])
        self.rounds = data['rounds']
        self.theta = data['theta']
        self.time = data['time']
        # Уменьшение здоровья, когда нет видимой причины
        if self.hp < self.hp_buf:
            self.self_damage = True

    def build_traj(self, behaivor: dict):
        if len(self.traj_stack) == self.max_stack:
            self.traj_stack = self.traj_stack[1:]
        self.traj_stack.append(behaivor)

    def get_behaivor(self) -> dict:
        # Основная функция "поведения" объекта
        if self.self_damage:
            self.rollback = True
            self.hp_buf = self.hp

        if self.rollback:
            self.__rollback_motion()
            return self.behaivor.copy()

        if not self.enemies:
            self.is_contact = False

            if self.was_contact:
                self.__searching_motion()
                self.steps += 1
                if self.steps > self.lim_steps:
                    self.was_contact = False
                    self.steps = 0
            else:
                self.__peace_motion()
        else:
            self.is_contact = True
            self.__contact_direction()
            self.was_contact = True

            self.__attack_motion() if self.enemies[0][1] > 0.1 * self.r_vis else self.__attack_motion(reverse=True)

        return self.behaivor.copy()

    def __peace_motion(self):
        # "Мирное" движение с обходом препятствий
        if self.rounds:
            self.__try_dodge()
            return
        # Сначала сканируем
        if self.timer < 40:
            self.__scanning_motion()
        elif self.timer < 150:
            # Логика поля видимости
            if self.switch == 0:
                if self.theta > 15:
                    self.__vision(-.5)
                else:
                    self.__vision(0)
                    self.switch = 1
            else:
                if self.theta < 90:
                    self.__vision(1)
                else:
                    self.__vision(0)
                    self.switch = 0
            # Логика для 'move' и 'rotate'
            if self.rays_intersected[self.midray] > self.min_range:
                self.__move(1)
                self.__rotate(.1)
            else:
                # Поиск луча, не упирающегося в препятствие
                max_i = np.argmax(self.rays_intersected)
                if self.rays_intersected[max_i] > self.min_range:
                    self.__move(1)
                    if max_i < self.midray:  # Луч с правого борта
                        self.__rotate(-1)
                    else:  # Луч с левого борта
                        self.__rotate(1)
                else:  # Нет луча полной длины -> просто разворачиваемся
                    self.__rotate(1)

            self.timer += self.dt
        else:
            self.timer = 0

    def __scanning_motion(self):
        # if not self.enemies:
        self.__move(.5)
        self.__rotate(1)
        self.__vision(-1)
        self.timer += self.dt

    def __attack_motion(self, reverse: bool = False):
        # Наведение на цель и стрельба
        for enemy in self.enemies:
            if enemy[1] < 1:
                self.was_contact = False
                self.out_of_vision = True
        # Логика поля зрения
        self.__vision(-1 if self.theta > 1 else 0)
        # Разворот на цель
        for enemy in self.enemies:
            self.__move(5 * enemy[1] / self.r_vis) if not reverse else self.__move(-5 * enemy[1] / self.r_vis)
            if enemy[0] > 1:
                self.__rotate(1)
            elif enemy[0] < -1:
                self.__rotate(-1)
            else:
                self.__may_fire(1)
                self.__rotate(0)

    def __rollback_motion(self):
        if self.traj_stack and self.traj_stack is not None:
            self.behaivor = self.traj_stack.pop().copy()
            self.behaivor['move'] *= -1
            self.behaivor['rotate'] *= -1
            self.behaivor['fire'] = 1
            self.behaivor['vision'] = 1 if self.theta < 15 else -1
        else:
            self.traj_stack = []
            if self.enemies:
                self.__attack_motion(reverse=True)
            elif self.sum_angle < 190:
                self.__circle_motion(with_move=False)
            else:
                self.sum_angle = 0
                self.rollback = False
                self.self_damage = False

    def __try_dodge(self):
        # Попытка уклонения от снаряда
        self.__move(1)
        for r in self.rounds:
            self.__rotate(-.9) if r[0] > 0 else self.__rotate(.9)

    def __searching_motion(self):
        self.__vision(-.75) if self.theta > 30 else self.__rotate(.75)
        self.__move(1)
        if self.alpha_to_enemy < self.alpha < 0 or 0 < self.alpha_to_enemy < self.alpha:
            self.__rotate(-1)
        elif 0 < self.alpha < self.alpha_to_enemy or self.alpha < self.alpha_to_enemy < 0:
            self.__rotate(1)

    def __circle_motion(self, with_move: bool = True):
        self.__move(1) if with_move else self.__move(0)
        self.__rotate(-1)
        self.sum_angle += 2         # Т.к. макс. скорость вращения 20 град/с
        self.__vision(-1) if self.theta > 30 else self.__vision(1)

    def __contact_direction(self):
        a = [self.alpha + enemy[0] for enemy in self.enemies]
        self.alpha_to_enemy = a[0]

    def __move(self, speed: float):
        self.behaivor['move'] = speed

    def __rotate(self, angle: float):
        self.behaivor['rotate'] = angle

    def __vision(self, vis: float):
        self.behaivor['vision'] = vis

    def __may_fire(self, val):
        self.behaivor['fire'] = val
