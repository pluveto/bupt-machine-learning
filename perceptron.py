import numpy as np

class MyPerceptron:
    def __init__(self) -> None:
        self.w = np.array([0, 0, 0])
        self.b = 0
        self.x = np.array([
            [0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0],  # 正实例点
            [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1],  # 负实例点
        ])
        self.y = [1, 1, 1, 1, -1, -1, -1, -1]
        self.history = [0, 0, 0, 0, 0, 0, 0, 0]
        self.step_cnt = 0
        self.ita = 1

    def cost(self, x_i, y_i):
        d = (np.dot(self.w, x_i) + self.b)
        return - y_i * d

    def show_args(self):
        print("w = " + str(self.w))
        print("b = " + str(self.b))

    def get_misclass(self):
        for i in range(len(self.x)):
            x_i = self.x[i]
            y_i = self.y[i]
            cost = self.cost(x_i, y_i)
            if(cost >= 0):
                self.history[i] = 1
                print("cost = " + str(cost))
                return i, None
        return None, True

    def step(self):
        '''训练一步'''
        print("step = " + str(self.step_cnt))
        i, err = self.get_misclass()
        if(err != None):
            print("finish")
            self.show_args()
            return

        print("i = " + str(i))
        x_i = self.x[i]
        y_i = self.y[i]
        print("x_i = " + str(x_i))
        print("y_i = " + str(y_i))
        self.w += self.ita * x_i * y_i
        self.b += self.ita * y_i
        self.step_cnt += 1
        self.show_args()
        print("")
        return True


p = MyPerceptron()
while p.step():
    pass
