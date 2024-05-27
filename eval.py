def eval(w, env, features, maxsteps):
    s = env.init()
    for l in range(maxsteps):
        s, c = env.step(s, (w@features(s)).argmin())
        if c: break
    return c, l + 1
