def load_golem_shot(filepath):
    """
    Robust CSV parser for GOLEM oscilloscope data.
    Returns: t_ms, Ip_A, Uloop_V
    """
    try:
        data = np.genfromtxt(filepath, delimiter=',',
                              skip_header=17,
                              usecols=(0, 1, 12))
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) > 1000:
            t_ms = data[:, 0] * 1000
            Ul   = data[:, 1]
            Ip   = np.abs(data[:, 2])
            return t_ms, Ip, Ul
    except Exception:
        pass
    # Manual fallback
    t_arr, Ip_arr, Ul_arr = [], [], []
    with open(filepath, errors='ignore') as f:
        lines = f.readlines()
    for line in lines[17:]:
        p = line.strip().split(',')
        if len(p) < 13:
            continue
        try:
            t_arr.append(float(p[0]) * 1000)
            Ul_arr.append(float(p[1]))
            Ip_arr.append(abs(float(p[12])))
        except (ValueError, IndexError):
            continue
    return np.array(t_arr), np.array(Ip_arr), np.array(Ul_arr)
