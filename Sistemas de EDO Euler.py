#Caso Unidimensional

def F(t, y):
    return (y)
    

def Euler(f, t0, y0, T, N):

    h = (T - t0)/N
    t = t0
    y = y0

    for i in range(0, N):
        y = y + h*f(t, y)
        t = t + h

    
    return(y)

#Caso Multidimensional

def F_mult(t, y):
    y1 = y[0]
    y2 = y[1]
    dy1 = y1
    dy2 = -y2
    return([dy1, dy2])

def g(h, v):
    novo_v = []

    for item in v:
        novo_v.append(item * h)

    return(novo_v)
    

def Euler_mult(f, t0, y0, T, N):

    h = (T - t0)/N
    t = t0
    y = y0

    for i in range(0, N):
        incremento = g(h, f(t, y))
        y = [y[j] + incremento[j] for j in range(len(y))]
        t = t + h

    return(y)
