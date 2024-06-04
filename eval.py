def eval(w, env, features, maxsteps):
    s = env.init()
    for h in range(maxsteps):
        s, c, done = env.step(s, (w@features(s)).argmin())
        if done: break
    return c, h

def mc_eval(w, env, features, maxsteps):
    s = env.init()
    # print('w', w.shape, w, 'features', features(s)[0].shape, features(s)[0])
    for h in range(maxsteps):
        s, c, done = env.step(s, (w@features(s)[0]).argmin(), h+1, maxsteps)
        if done: break
    return c[0], h
