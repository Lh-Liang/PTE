def stats_time(start, end, step, total_step):
    t = end -start
    return '{:.3f}'.format((t / step * (total_step - step) / 3600))


