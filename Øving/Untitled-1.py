def Lagrange(*args): pass #Dette er en funksjon som lager et Lagrange polynom med noder

def p(f, a, b, N, known):
    """
    f er funksjon
    a er start
    b er slutt
    N er antall kjente noder
    known er de kjente nodene med elementer på formen (x,f(x))
    """
    p.f, p.a, p.b, p.N, p.k = f, a, b, N, known


def cost(nodes):
        return (p.a-p.b)/p.N*sum([(p.f(k) - Lagrange([(x, p.f(x)) for x in nodes])(k))**2 for k,fk in p.known])

#Nå skal dette virke:
grad(cost)(nodes)