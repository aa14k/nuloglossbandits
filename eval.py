def eval(w, env, features, max, rounds):
    cs, lens = [], []
    for _ in range(rounds):
        s = env.init()
        for l in range(max):
            s, c = env.step(s, (w@features(s)).argmin())
            # print(s, c, (w@features(s)).argmin())
            if c: break
        cs.append(c), lens.append(l + 1)
    return cs, lens
