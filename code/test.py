from utils import ObsHolder

obs_holder:ObsHolder = ObsHolder.load("./saves/7")

print(obs_holder.get_kern_param())
