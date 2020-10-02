import numpy as np
import math
from typing import Sequence


def makeA(c: list):
    return -np.matrix([
        [1, 1, 0, c[0]],
        [1, 1, 0, - c[1]],
        [1, 0, 1, c[2]],
        [1, 0, 1, - c[3]],
        [1, -1, 0, c[4]],
        [1, -1, 0, - c[5]],
        [1, 0, -1, c[6]],
        [1, 0, -1, - c[7]],
    ])


def makeC(c: list):
    c0 = sum([c[j]*((-1)**j)for j in range(len(c))])
    c1 = c[0]-c[1]-c[4]+c[5]
    c2 = c[2]-c[3]-c[6]+c[7]
    return np.matrix([
        [8, 0, 0, c0],
        [0, 4, 0, c1],
        [0, 0, 4, c2],
        [c0, c1, c2, sum([i ** 2 for i in c])],
    ])


def makeOmega(theta: list):
    return np.matrix([
        [sum(theta) - 2 * math.pi],
        [theta[0] + theta[1] - theta[4] - theta[5]],
        [theta[2] + theta[3] - theta[6] - theta[7]],
        [sum([((-1)**j)*math.log(math.sin(theta[j]))
              for j in range(len(theta))])]
    ])


# 繰り返す函数
def calcResidual(theta: np.matrix):
    c = np.matrix([1.0/math.tan(*value) for value in theta.tolist()]).T
    A = makeA(*c.T.tolist())
    omega = makeOmega(*theta.T.tolist())
    C = makeC(*c.T.tolist())
    return A * (C ** -1) * omega

# 収束判定


def checkConvergence(residuals: np.matrix):
    if (len(residuals) == 0):
        return False
    value = max([max([math.fabs(residual) for residual in row])
                 for row in residuals])
    return value < 10.0**-9


def todegree(radian: float, digit: int = 10):
    degrees = math.degrees(radian)
    degf, deg = math.modf(degrees)
    minutes = int(degf*60)
    seconds = int(degf*3600) % 60+degf*3600-int(degf*3600)
    return f'{int(deg)}°{int(minutes)}′{seconds:.{digit}g}′′'


def printAngles(angles: Sequence[float], indent: int = 1, symbol: str = 'M'):
    space = ''.join(['\t']*indent)
    for i in range(len(angles)):
        print(f'{space}{symbol}_{i} = {todegree(angles[i],4)}')


def printTheta(angles: Sequence[float], step: int, indent: int = 1, symbol: str = 'θ'):
    space = ''.join(['\t']*indent)
    for i in range(len(angles)):
        print(f'{space}{symbol}_{i}({step}) = {todegree(angles[i],4)}')

    # main函数
if __name__ == '__main__':
    # 初期値
    theta = np.matrix([
        [0.757206248],
        [1.090661098],
        [0.682819669],
        [0.610873318],
        [0.720344248],
        [1.127482697],
        [0.821040049],
        [0.47273374],
    ], dtype=float)

    i = 0
    print('This program calculates angles in Radian')
    print('Measured angles:')
    printTheta(*theta.T.tolist(), i, 1)
    print('Start calculating...')
    # 残差を保持するlist
    residuals = []

    # 計算処理
    while not checkConvergence(residuals):
        residual = calcResidual(theta)
        residuals.append(*residual.T.tolist())
        theta += residual
        i += 1
        print(f'### result of step {i} ###')
        printTheta(*theta.T.tolist(), i)
        printTheta(residuals[0], i, symbol='⊿')
        if len(residuals) > 10:
            residuals.pop(0)

    print(f'##########################')
    print('Finish calculating!')
    print('Most probable angles:')
    printAngles(*theta.T.tolist())
    print('Residuals:')
    printAngles(residuals[len(residuals)-1], symbol='ν')
