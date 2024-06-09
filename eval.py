def eval(w, env, features, maxsteps):
    s = env.init()
    for h in range(maxsteps):
        s, c, done = env.step(s, (w@features(s)).argmin())
        if done: break
    return c, h

# def mc_eval(w, env, features, H):
#     s = env.init()
#     for h in range(H):
#         s, c = env.step(s, (w@features(s)).argmin())
#     return c[0]
