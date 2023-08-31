import math


def hypotenuse(x, y):
    x2 = x**2
    y2 = y**2
    z2 = x2 + y2
    the_result = math.sqrt(z2)
    return the_result

#print(hypotenuse(3, 4))


def cuboid_area(l, w, h):
    lw = l * w
    wh = w * h
    lh = l * h
    sa = 2 * (lw + wh + lh)
    return sa

print(cuboid_area(3, 5, 6))