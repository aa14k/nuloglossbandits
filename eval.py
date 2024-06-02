def eval(w, env, features, maxsteps):
    s = env.init()
    for l in range(maxsteps):
        s, c, done = env.step(s, (w@features(s)).argmin())
        if done: break
    return done, c
