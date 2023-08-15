from utils import ObsHolder, plot_mean

obs_holder:ObsHolder = ObsHolder.load("./saves/0")

print(obs_holder.get_kern_param())

plot_mean(obs_holder)
